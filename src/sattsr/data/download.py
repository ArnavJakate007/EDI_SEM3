"""Anonymous bulk download of GOES ABI files from the NOAA public S3 bucket.

Not covered by the test suite: it needs the network. Keep network access confined here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_BUCKET = "noaa-goes19"
DEFAULT_PRODUCT = "ABI-L1b-RadF"


def _filesystem():
    import s3fs

    return s3fs.S3FileSystem(anon=True)


def list_goes_keys(
    day: date,
    hours: Iterable[int],
    *,
    bucket: str = DEFAULT_BUCKET,
    product: str = DEFAULT_PRODUCT,
    channel: str = "C13",
) -> list[str]:
    """List the S3 keys for one day's channel files over the requested hours."""
    fs = _filesystem()
    keys: list[str] = []
    for hour in hours:
        prefix = f"{bucket}/{product}/{day.year}/{day.timetuple().tm_yday:03d}/{hour:02d}/"
        try:
            listing = fs.ls(prefix)
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
) -> list[Path]:
    """Download one day's Channel 13 full-disc files for the requested hours.

    `every` subsamples the listing (ABI full-disc is 10-minutely, so `every=3`
    yields a 30-minute cadence, which is the INSAT cadence this project targets).
    Files already present locally are not re-fetched.
    """
    fs = _filesystem()
    out_dir = Path(dest)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = list_goes_keys(day, hours, bucket=bucket, product=product, channel=channel)[::every]
    if max_files is not None:
        keys = keys[:max_files]

    if progress:
        from tqdm import tqdm

        keys = tqdm(keys, desc=f"GOES {day}")   # type: ignore[assignment]

    downloaded: list[Path] = []
    for key in keys:
        local = out_dir / Path(key).name
        if not local.exists() or local.stat().st_size == 0:
            log.info("downloading %s", key)
            fs.get(key, str(local))
        downloaded.append(local)
    return downloaded
