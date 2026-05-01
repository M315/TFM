"""
US Treasury par yield curve fetcher and term-structure interpolator.
Data: US Treasury XML feed (no API key required).
"""

import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from typing import Callable, Tuple

import numpy as np


_MATURITIES = {
    "BC_1MONTH":  1/12, "BC_2MONTH":  2/12, "BC_3MONTH":  3/12,
    "BC_4MONTH":  4/12, "BC_6MONTH":  6/12, "BC_1YEAR":   1.0,
    "BC_2YEAR":   2.0,  "BC_3YEAR":   3.0,  "BC_5YEAR":   5.0,
    "BC_7YEAR":   7.0,  "BC_10YEAR": 10.0,  "BC_20YEAR": 20.0,
    "BC_30YEAR": 30.0,
}


def fetch_treasury_curve(
    ref_date: date,
) -> Tuple[Callable[[float], float], date, np.ndarray, np.ndarray]:
    """
    Fetch the US Treasury par yield curve and return a rate interpolator.

    Returns (r, curve_date, Ts, rs) where r(T) is the continuously-compounded
    rate for maturity T (years), linearly interpolated between curve nodes.
    curve_date is the last available business day on or before ref_date.
    """
    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/"
        "interest-rates/pages/xml?data=daily_treasury_yield_curve"
        f"&field_tdr_date_value_month={ref_date.strftime('%Y%m')}"
    )
    with urllib.request.urlopen(url, timeout=15) as resp:
        root = ET.fromstring(resp.read())

    def local(tag: str) -> str:
        return tag.split("}")[-1]

    best_props, best_date = None, date.min
    for props in root.iter():
        if local(props.tag) != "properties":
            continue
        date_el = next((c for c in props if local(c.tag) == "NEW_DATE"), None)
        if date_el is None or not date_el.text:
            continue
        entry_date = date.fromisoformat(date_el.text[:10])
        if entry_date <= ref_date and entry_date > best_date:
            best_date, best_props = entry_date, props

    if best_props is None:
        raise ValueError(
            f"No Treasury curve data for {ref_date}. "
            "Try a date within a month with published data."
        )

    prop_map = {local(el.tag): el.text for el in best_props}
    Ts, rs = [], []
    for tag, T in _MATURITIES.items():
        val = prop_map.get(tag)
        if val:
            try:
                Ts.append(T)
                rs.append(np.log(1.0 + float(val) / 100.0))
            except ValueError:
                pass

    Ts = np.asarray(Ts)
    rs = np.asarray(rs)

    def r(T: float) -> float:
        return float(np.interp(T, Ts, rs))

    return r, best_date, Ts, rs
