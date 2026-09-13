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
#: ASSUMPTION -- these exact strings are NOT verified against a real product listing.
SCAN_MODE_PRODUCTS = {
    "routine": "3DIMG_L1C_ASIA_MER",
    "staggered": "3DIMG_L1C_ASIA_MER_STAGGERED",
    "rapid_scan": "3DIMG_L1C_SEC_RAPID",
}

#: What to look for in the portal if the exact product string above is absent.
_PRODUCT_HINTS = {
    "routine": "half-hourly full-frame Asia sector",
    "staggered": "staggered / interleaved 15-minute Asia sector",
    "rapid_scan": "rapid scan sector, sub-5-minute cadence",
}

#: Native cadence per scan mode, in minutes. Rapid scan is 4 min 30 s, not 4 min.
CADENCE_MINUTES = {"routine": 30.0, "staggered": 15.0, "rapid_scan": 4.5}

#: Which shipped config drives preprocessing for each mode.
_CONFIG_FOR = {
    "routine": "insat.yaml",
    "staggered": "insat_staggered.yaml",
    "rapid_scan": "insat_rapidscan.yaml",
}

#: The pattern InsatReader.timestamp_of actually applies, quoted in the instructions
#: so the two can never drift apart.
_TIME_PATTERN = r"_(?P<day>\d{2})(?P<mon>[A-Z]{3})(?P<year>\d{4})_(?P<hhmm>\d{4})"


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


def expected_filename_at(day: date, minutes_into_day: float, scan_mode: str) -> str:
    """Filename for the slot `minutes_into_day` after 06:00 on `day`.

    Used to show a second, consecutive example in the manual instructions, which is
    what makes the cadence concrete for whoever is downloading by hand.
    """
    total = 6 * 60 + int(round(minutes_into_day))
    return expected_filename(day, (total // 60) % 24, total % 60, scan_mode)


def manual_instructions(
    raw_root: str | Path, scan_mode: str, start: date, end: date
) -> str:
    """Step-by-step manual download guidance for one scan mode and date range."""
    dest = target_dir(raw_root, scan_mode)
    product = SCAN_MODE_PRODUCTS[scan_mode]
    sample = expected_filename(start, 6, 0, scan_mode)
    days = (end - start).days + 1

    cadence = CADENCE_MINUTES[scan_mode]
    per_day = int(round(24 * 60 / cadence))
    second_slot = expected_filename_at(start, cadence, scan_mode)

    return f"""
MANUAL DOWNLOAD -- INSAT-3DR {scan_mode} ({start} to {end}, {days} day(s))
{"=" * 74}

MOSDAC has no scriptable public API and ordering is an asynchronous, emailed-link
workflow, so fetch these by hand once. Follow every step literally -- step 8's
filename pattern is what build_manifest.py parses timestamps out of, and a renamed
file is silently dropped from the manifest rather than erroring.

  1. Sign in at {MOSDAC_BASE}.
     Registration is free but ACCOUNT APPROVAL IS MANUAL and can take 1-2 working
     days. Do this first; nothing else works until the account is active.

  2. Top menu: "Data Access" -> "Order Data"  (older builds label this
     "Open Data" -> "Order"). You must be signed in or the menu is not rendered.

  3. In the left-hand tree: Satellite -> INSAT-3DR -> IMAGER -> Level-1C.

  4. Product:                   {product}
     If that exact string is not offered, pick the L1C entry whose description
     mentions "{_PRODUCT_HINTS[scan_mode]}". See the ASSUMPTIONS note at the end.

  5. Date range. The selector is INCLUSIVE of both endpoints and works in whole
     UTC days -- there is no hour field, so you always receive complete days:
         From: {start:%d-%m-%Y}      To: {end:%d-%m-%Y}
     Expect roughly {per_day} files per day at this scan mode's {cadence:g}-minute
     cadence. Orders spanning more than ~7 days are frequently rejected for size;
     split a longer range into several orders.

  6. Region / sector:           ASIA_MER  (full Indian-region Mercator sector)
  7. Channel:                   leave ALL channels selected. The L1C product bundles
                                them in one .h5 and the reader picks IMG_TIR1 itself;
                                de-selecting channels sometimes yields a different
                                product layout.

  8. Place the order, wait for the email, and unpack every .h5 into EXACTLY:

         {dest.as_posix()}/

     KEEP MOSDAC'S ORIGINAL FILENAMES. The parser
     (sattsr.io.insat.InsatReader.timestamp_of) needs this substring:

         _DDMONYYYY_HHMM          regex:  {_TIME_PATTERN}

     so a correct name looks like:

         {sample}
     and the next slot that day is:
         {second_slot}

     MON is the 3-letter English month in CAPITALS (JAN FEB MAR APR MAY JUN JUL
     AUG SEP OCT NOV DEC). DD and HHMM are zero-padded. The month match is
     case-insensitive, but the day/month/year/HHMM ORDER must be intact. Do not
     rename, lowercase the date, replace it with an ISO date, or flatten the
     directory into per-day subfolders -- any of those makes the file invisible.

  9. VERIFY before moving on. This must report one row per file, 0 unparseable:

         python scripts/build_manifest.py --sensor insat3dr \\
             --raw-root {Path(raw_root).as_posix()} --scan-mode {scan_mode}

     If it says "excluded (timestamp not parseable)", the filenames are wrong --
     fix them rather than proceeding, or those frames are simply lost.

Then continue exactly as for the automated sensors:

    python scripts/preprocess.py --sensor insat3dr \\
        --config configs/{_CONFIG_FOR[scan_mode]} --workers 4

Nothing downstream knows or cares that these arrived by hand.

ASSUMPTIONS NOT YET VERIFIED (no MOSDAC account was available to check):
  * the product-stream name "{product}"
  * the LUT dataset name IMG_TIR1_TEMP inside the .h5
If either turns out to differ from what the portal actually offers, tell me the
real names and they are a one-line change each.
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
