"""Bulk download of raw satellite files from public archives.

Not covered by the test suite where it needs the network. Keep network access confined
here; the decision logic (retry policy, resumability) is factored into pure helpers
above so it can be tested without touching a socket.

S3 access uses `s3fs` with `anon=True` rather than `boto3` with an UNSIGNED signer.
The two are equivalent for anonymous public buckets, and `s3fs` is already declared in
the project's `download` extra and already installed, so this adds no dependency.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, TypeVar

log = logging.getLogger(__name__)

DEFAULT_BUCKET = "noaa-goes19"
DEFAULT_PRODUCT = "ABI-L1b-RadF"

#: NOAA's AWS Open Data mirrors for AHI. See `scripts/download_himawari8.py` for the
#: caveat about which Himawari product the repo's reader actually understands.
HIMAWARI_BUCKETS = {"himawari8": "noaa-himawari8", "himawari9": "noaa-himawari9"}
HIMAWARI_DEFAULT_PRODUCT = "AHI-L2-FLDK-ISatSS"

#: JAXA P-Tree gridded AHI, which is what `sattsr.io.himawari.HimawariReader` parses.
PTREE_HOST = "ftp.ptree.jaxa.jp"
PTREE_GRIDDED_ROOT = "/pub/himawari/L3/PAR/020"

T = TypeVar("T")

#: Errors worth retrying. Deliberately broad: s3fs/aiohttp raise a wide variety of
#: transport errors, and a failed retry is far cheaper than a failed overnight fetch.
TRANSIENT_ERRORS = (OSError, TimeoutError, ConnectionError)


@dataclass
class DownloadStats:
    """What a download pass actually did, for the final log line."""

    found: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    paths: list[Path] = field(default_factory=list)

    def log_summary(self, label: str) -> None:
        log.info(
            "%s: %d found in range, %d downloaded, %d skipped (already present), %d failed",
            label,
            self.found,
            self.downloaded,
            self.skipped,
            self.failed,
        )


def with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    description: str = "operation",
) -> T:
    """Run `fn`, retrying transient failures with exponential backoff.

    `sleep` is injectable so tests can exercise the backoff without waiting.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except TRANSIENT_ERRORS as exc:
            last = exc
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            log.warning(
                "%s failed (attempt %d/%d): %s -- retrying in %.1fs",
                description,
                attempt,
                attempts,
                exc,
                delay,
            )
            sleep(delay)

    assert last is not None
    raise last


def should_download(local: Path, remote_size: int | None = None) -> bool:
    """Decide whether `local` still needs fetching.

    Resumability is by name plus size: a file that is absent, empty, or a different
    size than the remote object is re-fetched. A partial file left behind by an
    interrupted run therefore heals itself on the next pass instead of being
    mistaken for a complete download.
    """
    if not local.exists():
        return True
    size = local.stat().st_size
    if size == 0:
        return True
    if remote_size is not None and size != remote_size:
        log.warning(
            "%s is %d bytes but remote is %d -- refetching", local.name, size, remote_size
        )
        return True
    return False


def _remote_size(fs: Any, key: str) -> int | None:
    """Best-effort object size; `None` when the listing does not carry one."""
    try:
        info = fs.info(key)
    except Exception:                                   # noqa: BLE001 - size is optional
        return None
    size = info.get("size") if isinstance(info, dict) else None
    return int(size) if isinstance(size, int) else None


def _filesystem():
    import s3fs

    return s3fs.S3FileSystem(anon=True)


def download_keys(
    fs: Any,
    keys: Sequence[str],
    out_dir: str | Path,
    *,
    progress: bool = False,
    label: str = "download",
    attempts: int = 3,
    verify_size: bool = True,
) -> DownloadStats:
    """Fetch every key into `out_dir`, skipping what is already complete.

    Individual failures are counted and logged rather than aborting the batch --
    partial archive availability is the norm and one missing hour should not sink
    an overnight fetch.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = DownloadStats(found=len(keys))

    iterator: Iterable[str] = keys
    if progress:
        from tqdm import tqdm

        iterator = tqdm(list(keys), desc=label)

    for key in iterator:
        local = out_dir / Path(key).name
        remote_size = _remote_size(fs, key) if verify_size else None

        if not should_download(local, remote_size):
            stats.skipped += 1
            stats.paths.append(local)
            continue

        try:
            with_retries(
                lambda k=key, dest=local: fs.get(k, str(dest)),
                attempts=attempts,
                description=f"fetch {Path(key).name}",
            )
        except TRANSIENT_ERRORS as exc:
            log.error("giving up on %s: %s", key, exc)
            stats.failed += 1
            continue

        log.info("downloaded %s", key)
        stats.downloaded += 1
        stats.paths.append(local)

    return stats


def list_goes_keys(
    day: date,
    hours: Iterable[int],
    *,
    bucket: str = DEFAULT_BUCKET,
    product: str = DEFAULT_PRODUCT,
    channel: str = "C13",
    fs: Any | None = None,
) -> list[str]:
    """List the S3 keys for one day's channel files over the requested hours."""
    fs = fs or _filesystem()
    keys: list[str] = []
    for hour in hours:
        prefix = f"{bucket}/{product}/{day.year}/{day.timetuple().tm_yday:03d}/{hour:02d}/"
        try:
            listing = with_retries(
                lambda p=prefix: fs.ls(p), description=f"list {prefix}"
            )
        except FileNotFoundError:
            log.warning("no data at %s", prefix)
            continue
        keys.extend(k for k in listing if channel in k and k.endswith(".nc"))
    return sorted(keys)


def fetch_goes_c13(
    dest: str | Path,
    *,
    day: date,
    hours: Iterable[int],
    bucket: str = DEFAULT_BUCKET,
    product: str = DEFAULT_PRODUCT,
    channel: str = "C13",
    max_files: int | None = None,
    every: int = 1,
    progress: bool = False,
    fs: Any | None = None,
) -> list[Path]:
    """Download one day's Channel 13 full-disc files for the requested hours.

    `every` subsamples the listing (ABI full-disc is 10-minutely, so `every=3`
    yields a 30-minute cadence, which is the INSAT cadence this project targets).
    Files already present and complete locally are not re-fetched.
    """
    fs = fs or _filesystem()
    keys = list_goes_keys(
        day, hours, bucket=bucket, product=product, channel=channel, fs=fs
    )[::every]
    if max_files is not None:
        keys = keys[:max_files]

    stats = download_keys(fs, keys, dest, progress=progress, label=f"GOES {day}")
    stats.log_summary(f"GOES-19 {day}")
    return stats.paths


def list_himawari_keys(
    day: date,
    hours: Iterable[int],
    *,
    bucket: str = HIMAWARI_BUCKETS["himawari8"],
    product: str = HIMAWARI_DEFAULT_PRODUCT,
    channel: int = 13,
    fs: Any | None = None,
) -> list[str]:
    """List AHI-L2-FLDK-ISatSS tile keys for one day's requested hours.

    Layout verified against the live bucket: `<bucket>/<product>/YYYY/MM/DD/HHMM/`,
    with each 10-minute slot holding 76 tiles per channel across all 16 channels.
    Tiles are named `OR_HFD-020-B12-M1C13-T074_GH8_s<YYYYDDDHHMMSS>_c....nc`, so the
    channel selector is `C<nn>-T`, not the `B13` token used by the HSD/L1b products.
    """
    fs = fs or _filesystem()
    token = f"C{int(channel):02d}-T"
    keys: list[str] = []
    for hour in hours:
        prefix = f"{bucket}/{product}/{day:%Y/%m/%d}/{hour:02d}"
        try:
            listing = with_retries(
                lambda p=prefix: fs.glob(f"{p}*/*"), description=f"list {prefix}"
            )
        except FileNotFoundError:
            log.warning("no data at %s", prefix)
            continue
        keys.extend(k for k in listing if token in k and k.endswith((".nc", ".nc4")))
    return sorted(keys)


def group_isatss_keys_by_slot(keys: Sequence[str]) -> dict[str, list[str]]:
    """Group tile keys by their `HHMM` slot directory, preserving order."""
    slots: dict[str, list[str]] = {}
    for key in keys:
        parts = key.split("/")
        slot = parts[-2] if len(parts) >= 2 else ""
        slots.setdefault(slot, []).append(key)
    return slots


def fetch_himawari_b13_aws(
    dest: str | Path,
    *,
    day: date,
    hours: Iterable[int],
    bucket: str = HIMAWARI_BUCKETS["himawari8"],
    product: str = HIMAWARI_DEFAULT_PRODUCT,
    channel: int = 13,
    max_files: int | None = None,
    max_slots: int | None = None,
    every: int = 1,
    progress: bool = False,
    fs: Any | None = None,
) -> DownloadStats:
    """Download ISatSS Band-13 tiles from the anonymous NOAA AWS mirror.

    `every` and `max_slots` subsample whole 10-minute SLOTS rather than individual
    files: a slot's 76 tiles are one frame, so dropping tiles at random would leave
    every frame with holes instead of giving fewer complete frames.
    """
    fs = fs or _filesystem()
    keys = list_himawari_keys(
        day, hours, bucket=bucket, product=product, channel=channel, fs=fs
    )
    slots = group_isatss_keys_by_slot(keys)
    ordered = sorted(slots)[::every]
    if max_slots is not None:
        ordered = ordered[:max_slots]

    selected = [k for slot in ordered for k in slots[slot]]
    if max_files is not None:
        selected = selected[:max_files]
    log.info(
        "%s %s: %d slot(s) selected of %d, %d tile(s) to fetch",
        product, day, len(ordered), len(slots), len(selected),
    )

    stats = download_keys(fs, selected, dest, progress=progress, label=f"Himawari {day}")
    stats.log_summary(f"Himawari AWS {day}")
    return stats


def fetch_himawari_b13_ptree(
    dest: str | Path,
    *,
    day: date,
    hours: Iterable[int],
    user: str,
    password: str,
    host: str = PTREE_HOST,
    root: str = PTREE_GRIDDED_ROOT,
    max_files: int | None = None,
    progress: bool = False,
) -> DownloadStats:
    """Download JAXA P-Tree gridded AHI files over FTP.

    This is the product `HimawariReader` understands (`NC_H08_<date>_<time>_...nc`
    carrying `tbb_13` in Kelvin). P-Tree requires a free registered account; pass it
    via `HIMAWARI_USER` / `HIMAWARI_PASS`. Uses stdlib `ftplib`, so no new dependency.
    """
    from ftplib import FTP

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    stats = DownloadStats()

    ftp = with_retries(lambda: FTP(host, timeout=60), description=f"connect {host}")
    try:
        ftp.login(user=user, passwd=password)
        for hour in hours:
            remote_dir = f"{root}/{day:%Y%m}/{day:%d}"
            try:
                names = with_retries(
                    lambda d=remote_dir: ftp.nlst(d), description=f"list {remote_dir}"
                )
            except TRANSIENT_ERRORS as exc:
                log.warning("cannot list %s: %s", remote_dir, exc)
                continue

            wanted = [
                n
                for n in names
                if f"_{day:%Y%m%d}_{hour:02d}" in Path(n).name and n.endswith(".nc")
            ]
            stats.found += len(wanted)

            iterator: Iterable[str] = wanted
            if progress:
                from tqdm import tqdm

                iterator = tqdm(wanted, desc=f"P-Tree {day} {hour:02d}z")

            for name in iterator:
                if max_files is not None and stats.downloaded >= max_files:
                    break
                local = dest / Path(name).name
                if not should_download(local):
                    stats.skipped += 1
                    stats.paths.append(local)
                    continue
                try:
                    with local.open("wb") as fh:
                        with_retries(
                            lambda n=name, h=fh: ftp.retrbinary(f"RETR {n}", h.write),
                            description=f"fetch {Path(name).name}",
                        )
                except TRANSIENT_ERRORS as exc:
                    log.error("giving up on %s: %s", name, exc)
                    local.unlink(missing_ok=True)
                    stats.failed += 1
                    continue
                stats.downloaded += 1
                stats.paths.append(local)
    finally:
        try:
            ftp.quit()
        except Exception:                               # noqa: BLE001 - best effort close
            ftp.close()

    stats.log_summary(f"Himawari P-Tree {day}")
    return stats
