"""Fetch KMA API Hub "2.1 단기예보" (short-term forecast grid) and reduce it to one location.

The API
-------
Endpoint: /api/typ01/cgi-bin/url/nph-dfs_shrt_grd on apihub.kma.go.kr
Parameters, and there are only four:

  tmfc  announcement time, YYYYMMDDHH (KST). Issued at 02/05/08/11/14/17/20/23 only; any other
        hour returns an all-missing grid rather than an error.
  tmef  effective time, YYYYMMDDHH (KST), hourly, at or after tmfc.
  vars  ONE forecast variable: TMP TMX TMN UUU VVV VEC WSD SKY PTY POP PCP SNO REH WAV.
  authKey

There is NO region parameter. This is a grid service: one call returns the whole national grid
for one variable at one (tmfc, tmef), and a place is selected out of the RESPONSE. (`reg`
belongs to the separate 구역별 API, fct_afs_dl.php, and is not accepted here.)

Response format, established by inspection rather than from the documentation
-----------------------------------------------------------------------------
A bare comma-separated array of floats, 20 per line, no header and no coordinates. A live call
returned exactly 37,697 values = 149 x 253, the 동네예보 grid, of which 9,295 were not -99.00
(the missing marker for sea and out-of-domain cells).

Row order runs ny ASCENDING, so index (ny-1)*149 + (nx-1). Checked by lookup rather than
assumed: read the other way, Taebaek, Busan and Seogwipo all land on missing cells, while
ascending gives 16.0 / 24.0 / 23.0 degC for an August midnight. Seoul cannot settle the
question -- 253-127+1 = 127 maps to itself.

Locating a place
----------------
`latlon_to_grid` is the KMA's published Lambert conformal conic conversion (dfs_xy_conv), not
an approximation of it. Cross-checked against the published reference point: Seoul City Hall
(37.5665, 126.9780) -> (60, 127), which is the coordinate KMA's own documentation uses.
Taebaek City Hall (37.1641, 128.9856) -> (95, 119).

Still unconfirmed (marked rather than guessed): how far ahead of tmfc a tmef may be, and
whether `vars` accepts several variables at once. The documented example passes exactly one, so
this script issues one request per variable and concatenates -- correct either way.

Usage:
    cp .env.example .env && edit it            # KMA_API_KEY, KMA_API_URL
    set -a && . ./.env && set +a
    python3 scripts/fetch_kma_shrt_grd.py --tmfc 2026081117 --tmef 2026081200
    python3 scripts/fetch_kma_shrt_grd.py --tmfc 2026081117 --tmef 2026081200 --out taebaek.csv
    python3 scripts/fetch_kma_shrt_grd.py --tmfc 2026081117 --tmef 2026081200 --raw --vars TMP
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# BASE path only -- never a query string, never the key. The documented sample URL carries
# tmfc/tmef/vars/authKey as an example; pasting it whole would commit the key and send every
# parameter twice. `normalise_url` rejects that rather than letting it pass silently.
API_URL = os.environ.get(
    "KMA_API_URL", "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd"
)
API_KEY = os.environ.get("KMA_API_KEY", "")

# 149 (nx) x 253 (ny), the standard 동네예보 grid; confirmed against the response length.
GRID_NX, GRID_NY = 149, 253
MISSING = -90.0  # the API writes -99.00; anything below this is missing

# Taebaek City Hall through latlon_to_grid (see module docstring).
TAEBAEK_NX, TAEBAEK_NY = 95, 119

ALL_VARS = ["TMP", "TMX", "TMN", "UUU", "VVV", "VEC", "WSD", "SKY", "PTY",
            "POP", "PCP", "SNO", "REH", "WAV"]
VALID_TMFC_HOURS = {2, 5, 8, 11, 14, 17, 20, 23}


def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    """KMA's published dfs_xy_conv: WGS84 degrees -> (nx, ny) on the 5 km 동네예보 grid."""
    RE, GRID = 6371.00877, 5.0            # earth radius (km), grid spacing (km)
    SLAT1, SLAT2 = 30.0, 60.0             # standard parallels
    OLON, OLAT = 126.0, 38.0              # origin
    XO, YO = 43, 136                      # origin in grid units
    DEGRAD = math.pi / 180.0

    re = RE / GRID
    slat1, slat2 = SLAT1 * DEGRAD, SLAT2 * DEGRAD
    olon, olat = OLON * DEGRAD, OLAT * DEGRAD

    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = sf ** sn * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / ro ** sn

    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / ra ** sn
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn
    return int(ra * math.sin(theta) + XO + 0.5), int(ro - ra * math.cos(theta) + YO + 0.5)


def normalise_url(api_url: str) -> str:
    """Strip a query string, and refuse a key embedded in the URL."""
    base, separator, query = api_url.partition("?")
    if separator and "authkey" in query.lower():
        raise SystemExit(
            "The API URL contains an authKey. Remove it and pass the key through KMA_API_KEY. "
            "If that URL was ever committed, treat the key as exposed and reissue it."
        )
    if separator:
        print(f"ignoring query string on the API URL: {query!r}", file=sys.stderr)
    return base


def check_times(tmfc: str, tmef: str) -> None:
    for name, value in (("tmfc", tmfc), ("tmef", tmef)):
        if len(value) != 10 or not value.isdigit():
            raise SystemExit(f"{name} must be YYYYMMDDHH, got {value!r}")
    if int(tmfc[8:10]) not in VALID_TMFC_HOURS:
        raise SystemExit(
            f"tmfc hour {tmfc[8:10]} is not an issuance hour {sorted(VALID_TMFC_HOURS)}; "
            "the API answers with an all-missing grid for other hours"
        )
    if tmef < tmfc:
        raise SystemExit(f"tmef ({tmef}) is before tmfc ({tmfc})")


def fetch(api_url: str, api_key: str, tmfc: str, tmef: str, var: str, timeout: int = 60) -> str:
    params = {"tmfc": tmfc, "tmef": tmef, "vars": var, "authKey": api_key}
    response = requests.get(api_url, params=params, timeout=timeout)
    response.raise_for_status()
    body = response.text
    # A bad key or bad time still comes back as HTTP 200 with a text body, so the status code
    # alone proves nothing.
    if not body.strip() or "error" in body[:400].lower() or "인증" in body[:400]:
        raise RuntimeError(f"request for {var} returned no usable data:\n{body[:400]}")
    return body


def parse_grid(body: str) -> np.ndarray:
    """Response text -> (ny, nx) array with missing cells as NaN."""
    values = [v.strip() for v in body.replace("\n", ",").split(",") if v.strip()]
    array = np.array(values, dtype=float)
    expected = GRID_NX * GRID_NY
    if array.size != expected:
        raise ValueError(
            f"expected {expected} values ({GRID_NX}x{GRID_NY}) but parsed {array.size}; "
            "the response layout may have changed -- rerun with --raw"
        )
    grid = array.reshape(GRID_NY, GRID_NX)
    grid[grid < MISSING] = np.nan
    return grid


def value_at(grid: np.ndarray, nx: int, ny: int) -> float:
    """Grid value at a 1-based (nx, ny). Rows run ny ascending (see module docstring)."""
    if not (1 <= nx <= GRID_NX and 1 <= ny <= GRID_NY):
        raise SystemExit(f"(nx, ny) = ({nx}, {ny}) is outside the {GRID_NX}x{GRID_NY} grid")
    return float(grid[ny - 1, nx - 1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--tmfc", required=True, help="announcement time YYYYMMDDHH (KST)")
    parser.add_argument("--tmef", required=True, help="effective time YYYYMMDDHH (KST)")
    parser.add_argument("--vars", nargs="+", default=ALL_VARS)
    parser.add_argument("--nx", type=int, default=TAEBAEK_NX)
    parser.add_argument("--ny", type=int, default=TAEBAEK_NY)
    parser.add_argument("--latlon", nargs=2, type=float, metavar=("LAT", "LON"),
                        help="use this coordinate instead of --nx/--ny")
    parser.add_argument("--raw", action="store_true", help="print the head of each response and stop")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if not args.api_url:
        raise SystemExit("no API URL: pass --api-url or set KMA_API_URL")
    if not API_KEY:
        raise SystemExit("no API key: set KMA_API_KEY in the environment")
    api_url = normalise_url(args.api_url)
    check_times(args.tmfc, args.tmef)

    nx, ny = (latlon_to_grid(*args.latlon) if args.latlon else (args.nx, args.ny))
    if not args.raw:
        print(f"grid cell (nx, ny) = ({nx}, {ny})")

    rows = []
    for var in args.vars:
        body = fetch(api_url, API_KEY, args.tmfc, args.tmef, var)
        if args.raw:
            print(f"===== {var} =====")
            print("\n".join(body.splitlines()[:5]))
            continue
        grid = parse_grid(body)
        rows.append({
            "tmfc": args.tmfc,
            "tmef": args.tmef,
            "nx": nx,
            "ny": ny,
            "var": var,
            "value": value_at(grid, nx, ny),
            "valid_cells": int(np.isfinite(grid).sum()),
        })

    if args.raw:
        return

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    if args.out:
        frame.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
