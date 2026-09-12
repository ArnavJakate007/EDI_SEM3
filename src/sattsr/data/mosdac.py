"""MOSDAC (ISRO) session downloader for INSAT-3DR/3DS, plus a manual fallback.

MOSDAC has no stable, documented, anonymous API. It is a server-rendered portal behind
a login form, and both the form's field names and the order-placement flow have changed
between revisions. Anything hard-coded here is a guess about markup that may already
have moved, so this module is written to *fail loudly and tell you exactly what to do
by hand* rather than to silently return zero files and let Stage 2 look merely empty.

`MosdacSession` is therefore best-effort and unverified against the live portal. The
manual path via `manual_instructions()` is the supported one until the endpoints below
are confirmed. Everything downstream -- `scripts/build_manifest.py`, `preprocess.py`,
the dataset and the training loop -- reads only from the local directory layout, so
manually-placed files work identically to downloaded ones.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

MOSDAC_BASE = "https://www.mosdac.gov.in"
LOGIN_PATH = "/internal/login"
ORDER_PATH = "/internal/order-data"

#: Env vars carrying MOSDAC credentials. Never hard-code or log these.
USER_ENV = "MOSDAC_USER"
PASS_ENV = "MOSDAC_PASS"

#: MOSDAC ships each scan mode as a distinct product stream.
SCAN_MODE_PRODUCTS = {
    "routine": "3DIMG_L1C_ASIA_MER",
    "staggered": "3DIMG_L1C_ASIA_MER_STAGGERED",
    "rapid_scan": "3DIMG_L1C_SEC_RAPID",
}


class MosdacError(RuntimeError):
    """Raised when MOSDAC cannot be reached, authenticated, or parsed."""


@dataclass(frozen=True)
class MosdacCredentials:
    """Credentials pulled from the environment."""

    user: str
    password: str

    @classmethod
    def from_env(cls) -> MosdacCredentials:
        """Read credentials, raising a message that says exactly what to set."""
        user, password = os.environ.get(USER_ENV), os.environ.get(PASS_ENV)
        if not user or not password:
            raise MosdacError(
                f"MOSDAC credentials not found. Set {USER_ENV} and {PASS_ENV} "
                f"(register free at {MOSDAC_BASE}), or use the manual download "
                f"path -- run scripts/download_insat3dr.py --manual-instructions."
            )
        return cls(user=user, password=password)


def target_dir(raw_root: str | Path, scan_mode: str) -> Path:
    """`<raw_root>/<scan_mode>/` -- the layout the manifest builder infers modes from."""
    if scan_mode not in SCAN_MODE_PRODUCTS:
        raise ValueError(
            f"unknown scan mode {scan_mode!r}; expected one of {sorted(SCAN_MODE_PRODUCTS)}"
        )
    return Path(raw_root) / scan_mode


def expected_filename(day: date, hour: int, minute: int, scan_mode: str) -> str:
    """The MOSDAC L1C filename `InsatReader.timestamp_of` expects for this slot.

    Kept in one place so the manual instructions and any future automated fetch
    cannot drift apart from the reader's `_DDMONYYYY_HHMM` parser.
    """
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    stamp = f"{day.day:02d}{months[day.month - 1]}{day.year}_{hour:02d}{minute:02d}"
    suffix = "SEC_RAPID" if scan_mode == "rapid_scan" else "ASIA_MER"
    return f"3DIMG_{stamp}_L1C_{suffix}.h5"


def manual_instructions(
    raw_root: str | Path, scan_mode: str, start: date, end: date
) -> str:
    """Step-by-step manual download guidance for one scan mode and date range."""
    dest = target_dir(raw_root, scan_mode)
    product = SCAN_MODE_PRODUCTS[scan_mode]
    sample = expected_filename(start, 6, 0, scan_mode)
    days = (end - start).days + 1

    return f"""
MANUAL DOWNLOAD -- INSAT-3DR {scan_mode} ({start} to {end}, {days} day(s))
{"=" * 72}

MOSDAC has no scriptable public API, so fetch these by hand once:

  1. Sign in at {MOSDAC_BASE} (free registration; approval can take a day or two).
  2. Open "Order Data" -> "Open Data" -> INSAT-3DR -> IMAGER.
  3. Select product:            {product}
  4. Set the date range:        {start} to {end}
  5. Region:                    ASIA_MER (full Indian-region mercator sector)
  6. Channel:                   TIR1 (10.8 um) -- the L1C product bundles all
                                channels, which is fine; the reader picks IMG_TIR1.
  7. Place the order. MOSDAC emails a download link, typically within a few hours.
  8. Unpack every .h5 into EXACTLY this directory (create it if needed):

         {dest.as_posix()}/

     Keep MOSDAC's original filenames. They must match the pattern
     `_DDMONYYYY_HHMM` that sattsr.io.insat.InsatReader.timestamp_of parses, e.g.

         {sample}

     Do NOT rename, lowercase, or flatten the date -- the parser is case-insensitive
     on the month but needs the day/month/year/HHMM layout intact.

Then continue exactly as for the automated sensors:

    python scripts/build_manifest.py --sensor insat3dr --raw-root {Path(raw_root).as_posix()}
    python scripts/preprocess.py --sensor insat3dr --workers 4

Nothing downstream knows or cares that these arrived by hand.
""".strip()


class MosdacSession:
    """Best-effort authenticated MOSDAC session. Unverified against the live portal.

    Every method raises `MosdacError` with actionable text rather than returning
    empty results, so a silent no-op can never be mistaken for "no data available".
    """

    def __init__(self, credentials: MosdacCredentials, *, base: str = MOSDAC_BASE) -> None:
        try:
            import requests
        except ImportError as exc:                      # pragma: no cover - env dependent
            raise MosdacError(
                "the 'requests' package is required for MOSDAC access; "
                'install the download extra:  pip install -e ".[download]"'
            ) from exc

        self._requests = requests
        self.base = base.rstrip("/")
        self.credentials = credentials
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "sattsr/0.1 (research use)"})

    def login(self) -> None:
        """Authenticate. Raises with guidance if the form has moved or changed."""
        url = f"{self.base}{LOGIN_PATH}"
        try:
            response = self.session.post(
                url,
                data={
                    "username": self.credentials.user,
                    "password": self.credentials.password,
                },
                timeout=60,
                allow_redirects=True,
            )
        except Exception as exc:                        # noqa: BLE001 - network surface
            raise MosdacError(f"could not reach MOSDAC at {url}: {exc}") from exc

        if response.status_code >= 400:
            raise MosdacError(
                f"MOSDAC login returned HTTP {response.status_code}. The login form at "
                f"{url} has most likely changed. Use the manual path: "
                f"scripts/download_insat3dr.py --manual-instructions"
            )
        if "logout" not in response.text.lower():
            raise MosdacError(
                "MOSDAC login did not appear to succeed (no logout link in the response). "
                "Either the credentials are wrong or the form fields are no longer "
                "'username'/'password'. Use --manual-instructions."
            )
        log.info("authenticated to MOSDAC as %s", self.credentials.user)

    def fetch_range(
        self, scan_mode: str, start: date, end: date, raw_root: str | Path
    ) -> list[Path]:
        """Order and download a date range. Currently always raises with guidance."""
        dest = target_dir(raw_root, scan_mode)
        dest.mkdir(parents=True, exist_ok=True)
        raise MosdacError(
            "MOSDAC's order-placement flow cannot be reliably scripted without "
            f"confirmed endpoint details for {self.base}{ORDER_PATH} -- it is an "
            "asynchronous, emailed-link workflow behind server-rendered forms. "
            "Run with --manual-instructions and follow those steps; everything "
            f"downstream reads from {dest.as_posix()}/ and works identically."
        )


def date_range(start: date, end: date) -> list[date]:
    """Inclusive list of dates from `start` to `end`."""
    if end < start:
        raise ValueError(f"end {end} precedes start {start}")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]
