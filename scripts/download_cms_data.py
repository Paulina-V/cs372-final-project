"""Download and convert CMS 2026 Medicare fee schedule data into the
project's cms_fee_schedule.csv format.

Combines CMS public fee schedule files and public reference tables:

1. Physician Fee Schedule (PFS) — procedure codes priced by RVU × conversion factor
   Source: https://pfs.data.cms.gov/
   File:   https://pfs.data.cms.gov/sites/default/files/data/indicators2026-03-18-2026.csv

2. Clinical Laboratory Fee Schedule (CLFS) — lab/pathology codes with national rates
   Source: https://www.cms.gov/medicare/payment/fee-schedules/clinical-laboratory-fee-schedule-clfs/files
   File:   https://www.cms.gov/files/zip/26clabq1.zip

3. Anesthesia codes — base units from CMS nationwide table × national conversion factor
   Base units source: https://www.va.gov/COMMUNITYCARE/docs/RO/Outpatient-DataTables/v3-27_Table-H.pdf
   Conversion factor source: https://www.cms.gov/files/zip/2026-anesthesia-conversion-factors.zip
   2026 national CF: $20.4976 (non-qualifying APM)

CPT descriptions may be subject to AMA licensing terms; use this script
and derived data for educational purposes.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "cms_fee_schedule.csv"

# --- PFS (Physician Fee Schedule) ---
PFS_RAW_PATH = ROOT / "data" / "cms_indicators_2026_raw.csv"
PFS_URL = (
    "https://pfs.data.cms.gov/sites/default/files/data/"
    "indicators2026-03-18-2026.csv"
)

# --- CLFS (Clinical Laboratory Fee Schedule) ---
CLFS_RAW_PATH = ROOT / "data" / "clfs_2026_q1.csv"
CLFS_URL = "https://www.cms.gov/files/zip/26clabq1.zip"

# --- Anesthesia ---
ANES_CF_PATH = ROOT / "data" / "anes_cf_2026.csv"
ANES_CF_URL = "https://www.cms.gov/files/zip/2026-anesthesia-conversion-factors.zip"
ANES_NATIONAL_CF = 20.4976  # 2026 non-qualifying APM national CF

# Base units extracted from VA nationwide table (CMS base units, stable across years).
# Source: https://www.va.gov/COMMUNITYCARE/docs/RO/Outpatient-DataTables/v3-27_Table-H.pdf
ANES_BASE_UNITS: dict[str, float] = {
    "00100": 5.0, "00102": 6.0, "00103": 5.0, "00104": 4.0, "00120": 5.0,
    "00124": 4.0, "00126": 4.0, "00140": 5.0, "00142": 4.0, "00144": 6.0,
    "00145": 6.0, "00147": 4.0, "00148": 4.0, "00160": 5.0, "00162": 7.0,
    "00164": 4.0, "00170": 5.0, "00172": 6.0, "00174": 6.0, "00176": 7.0,
    "00190": 5.0, "00192": 7.0, "00210": 11.0, "00211": 10.0, "00212": 5.0,
    "00214": 9.0, "00215": 9.0, "00216": 15.0, "00218": 13.0, "00220": 10.0,
    "00222": 6.0, "00300": 5.0, "00320": 6.0, "00322": 6.0, "00326": 7.0,
    "00350": 5.0, "00352": 5.0, "00400": 3.0, "00402": 3.0, "00404": 6.0,
    "00406": 6.0, "00410": 3.0, "00450": 5.0, "00454": 4.0, "00470": 5.0,
    "00472": 5.0, "00474": 6.0, "00500": 5.0, "00520": 13.0, "00522": 6.0,
    "00524": 4.0, "00528": 8.0, "00529": 6.0, "00530": 6.0, "00532": 4.0,
    "00534": 7.0, "00537": 7.0, "00539": 18.0, "00540": 12.0, "00541": 15.0,
    "00542": 15.0, "00546": 15.0, "00548": 17.0, "00550": 10.0, "00560": 15.0,
    "00561": 25.0, "00562": 20.0, "00563": 25.0, "00566": 25.0, "00567": 18.0,
    "00580": 20.0, "00600": 10.0, "00604": 13.0, "00620": 10.0, "00625": 13.0,
    "00626": 15.0, "00630": 8.0, "00632": 7.0, "00635": 4.0, "00640": 3.0,
    "00670": 13.0, "00700": 4.0, "00702": 4.0, "00730": 5.0, "00731": 5.0,
    "00732": 6.0, "00750": 4.0, "00752": 6.0, "00754": 4.0, "00756": 7.0,
    "00770": 8.0, "00790": 7.0, "00792": 7.0, "00794": 7.0, "00796": 10.0,
    "00797": 9.0, "00800": 6.0, "00802": 7.0, "00810": 5.0, "00820": 6.0,
    "00830": 5.0, "00832": 7.0, "00834": 7.0, "00836": 7.0, "00840": 5.0,
    "00842": 7.0, "00844": 7.0, "00846": 8.0, "00848": 8.0, "00851": 6.0,
    "00860": 6.0, "00862": 7.0, "00864": 8.0, "00865": 7.0, "00866": 10.0,
    "00868": 10.0, "00870": 5.0, "00872": 7.0, "00873": 5.0, "00880": 15.0,
    "00882": 10.0, "00902": 5.0, "00904": 7.0, "00906": 4.0, "00908": 6.0,
    "00910": 3.0, "00912": 5.0, "00914": 5.0, "00916": 5.0, "00918": 5.0,
    "00920": 3.0, "00921": 3.0, "00922": 6.0, "00924": 4.0, "00926": 4.0,
    "00928": 6.0, "00930": 4.0, "00932": 4.0, "00934": 6.0, "00936": 6.0,
    "00938": 6.0, "00940": 3.0, "00942": 5.0, "00944": 8.0, "00948": 4.0,
    "00950": 5.0, "00952": 5.0, "01112": 4.0, "01120": 5.0, "01130": 4.0,
    "01140": 4.0, "01150": 4.0, "01160": 3.0, "01170": 3.0, "01173": 6.0,
    "01180": 5.0, "01190": 3.0, "01200": 4.0, "01202": 7.0, "01210": 5.0,
    "01212": 5.0, "01214": 8.0, "01215": 8.0, "01220": 7.0, "01230": 3.0,
    "01232": 5.0, "01234": 4.0, "01250": 4.0, "01260": 7.0, "01270": 5.0,
    "01272": 5.0, "01274": 6.0, "01320": 5.0, "01340": 3.0, "01360": 3.0,
    "01380": 3.0, "01382": 3.0, "01390": 3.0, "01392": 7.0, "01400": 4.0,
    "01402": 7.0, "01404": 4.0, "01420": 3.0, "01430": 3.0, "01432": 3.0,
    "01440": 4.0, "01442": 7.0, "01444": 4.0, "01462": 3.0, "01464": 4.0,
    "01470": 3.0, "01472": 3.0, "01474": 4.0, "01480": 3.0, "01482": 5.0,
    "01484": 4.0, "01486": 5.0, "01490": 3.0, "01500": 4.0, "01502": 6.0,
    "01520": 5.0, "01522": 8.0, "01610": 3.0, "01620": 4.0, "01622": 4.0,
    "01630": 5.0, "01634": 5.0, "01636": 5.0, "01638": 5.0, "01650": 3.0,
    "01652": 4.0, "01654": 4.0, "01656": 4.0, "01670": 3.0, "01680": 5.0,
    "01710": 3.0, "01712": 5.0, "01714": 4.0, "01716": 7.0, "01730": 3.0,
    "01732": 5.0, "01740": 4.0, "01742": 5.0, "01744": 6.0, "01756": 3.0,
    "01758": 3.0, "01760": 3.0, "01770": 6.0, "01772": 6.0, "01780": 5.0,
    "01782": 5.0, "01810": 3.0, "01820": 3.0, "01829": 3.0, "01830": 3.0,
    "01832": 5.0, "01840": 4.0, "01842": 5.0, "01844": 4.0, "01850": 3.0,
    "01852": 4.0, "01860": 7.0, "01916": 5.0, "01920": 7.0, "01922": 7.0,
    "01924": 10.0, "01925": 5.0, "01926": 7.0, "01930": 5.0, "01931": 3.0,
    "01932": 3.0, "01933": 7.0, "01935": 5.0, "01936": 5.0, "01951": 3.0,
    "01952": 5.0, "01953": 3.0, "01958": 7.0, "01960": 7.0, "01961": 8.0,
    "01962": 5.0, "01963": 10.0, "01965": 5.0, "01966": 7.0, "01967": 5.0,
    "01968": 8.0, "01969": 9.0, "01990": 7.0, "01991": 3.0, "01992": 5.0,
    "01996": 1.0, "01999": 3.0,
}


def _download_if_missing(url: str, path: Path, label: str) -> None:
    if path.exists():
        print(f"{label} already cached at {path}")
        return
    print(f"Downloading {label} from {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}")


def _download_and_unzip_csv(url: str, path: Path, label: str) -> None:
    """Download a ZIP, extract the first CSV inside, save to path."""
    if path.exists():
        print(f"{label} already cached at {path}")
        return
    print(f"Downloading {label} from {url} ...")
    resp = urllib.request.urlopen(url)
    data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found in {url}")
        chosen = csv_names[0]
        print(f"  Extracting {chosen} ...")
        path.write_bytes(zf.read(chosen))
    print(f"Saved to {path}")


def download_all():
    """Download all raw data files."""
    _download_if_missing(PFS_URL, PFS_RAW_PATH, "CMS 2026 PFS indicators")
    _download_and_unzip_csv(CLFS_URL, CLFS_RAW_PATH, "CMS 2026 CLFS Q1")


def _load_pfs(seen: dict[str, dict]) -> int:
    """Load Physician Fee Schedule codes.

    Fee = full_nfac_total (total non-facility RVUs) × conv_fact.
    Only active codes (proc_stat == 'A') with a base modifier (blank) and
    a positive fee are included.
    """
    count = 0
    with PFS_RAW_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["proc_stat"].strip() != "A":
                continue
            if r["modifier"].strip():
                continue

            code = r["hcpc"].strip()
            rvu_total = float(r["full_nfac_total"] or 0)
            conv = float(r["conv_fact"] or 0)
            fee = round(rvu_total * conv, 2)
            if fee <= 0:
                continue

            if code not in seen or fee > seen[code]["fee"]:
                seen[code] = {
                    "code": code,
                    "description": r["sdesc"].strip(),
                    "fee": fee,
                }
                count += 1
    print(f"  PFS: {count} codes loaded")
    return count


def _load_clfs(seen: dict[str, dict]) -> int:
    """Load Clinical Laboratory Fee Schedule codes.

    The CLFS CSV has 4 header lines before the DictReader row.
    Only codes with a positive national rate and no modifier are included.
    CLFS codes only added if not already present from PFS (PFS takes priority).
    """
    count = 0
    with CLFS_RAW_PATH.open(newline="", encoding="latin-1") as f:
        # Skip 4 preamble lines
        for _ in range(4):
            next(f)
        reader = csv.DictReader(f)
        for r in reader:
            code = (r.get("HCPCS") or "").strip()
            rate_str = (r.get("RATE") or "").strip()
            mod = (r.get("MOD") or "").strip()
            desc = (r.get("SHORTDESC") or "").strip()

            if not code or not rate_str or mod:
                continue

            rate = float(rate_str)
            if rate <= 0:
                continue

            if code not in seen or rate > seen[code]["fee"]:
                seen[code] = {"code": code, "description": desc, "fee": rate}
                count += 1
    print(f"  CLFS: {count} codes loaded")
    return count


def _load_anesthesia(seen: dict[str, dict]) -> int:
    """Load anesthesia codes using base units × national conversion factor.

    Anesthesia codes (00100-01999) are in the PFS raw data with proc_stat='J'
    but have $0 RVUs. Their fee is: base_units × conversion_factor.
    Descriptions come from the PFS raw file.
    """
    # Grab descriptions from the PFS raw data
    descriptions: dict[str, str] = {}
    with PFS_RAW_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            code = r["hcpc"].strip()
            if code in ANES_BASE_UNITS and not r["modifier"].strip():
                descriptions[code] = r["sdesc"].strip()

    count = 0
    for code, base_units in ANES_BASE_UNITS.items():
        fee = round(base_units * ANES_NATIONAL_CF, 2)
        if fee <= 0:
            continue
        desc = descriptions.get(code, f"Anesthesia {code}")
        if code not in seen:
            seen[code] = {"code": code, "description": desc, "fee": fee}
            count += 1
    print(f"  Anesthesia: {count} codes loaded")
    return count


def convert():
    """Merge all fee schedule sources and write the combined CSV."""
    seen: dict[str, dict] = {}

    _load_pfs(seen)
    _load_clfs(seen)
    _load_anesthesia(seen)

    out_rows = sorted(seen.values(), key=lambda x: x["code"])

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "description", "fee"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} total codes to {OUT_PATH}")


if __name__ == "__main__":
    download_all()
    convert()
