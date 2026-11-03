"""
Data preprocessing utilities for claims, providers, and beneficiaries.
Standardizes data types, formats clinical codes, and parses demographic information.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from config.settings import USD_TO_INR_EFFECTIVE_DATE, USD_TO_INR_RATE


# State FIPS code mappings for Medicare Data
STATE_MAPPINGS = {
    1: "Alabama", 2: "Alaska", 3: "Arizona", 4: "Arkansas", 5: "California",
    6: "Colorado", 7: "Connecticut", 8: "Delaware", 9: "District of Columbia",
    10: "Florida", 11: "Georgia", 12: "Hawaii", 13: "Idaho", 14: "Illinois",
    15: "Indiana", 16: "Iowa", 17: "Kansas", 18: "Kentucky", 19: "Louisiana",
    20: "Maine", 21: "Maryland", 22: "Massachusetts", 23: "Michigan", 24: "Minnesota",
    25: "Mississippi", 26: "Missouri", 27: "Montana", 28: "Nebraska", 29: "Nevada",
    30: "New Hampshire", 31: "New Jersey", 32: "New Mexico", 33: "New York",
    34: "North Carolina", 35: "North Dakota", 36: "Ohio", 37: "Oklahoma", 38: "Oregon",
    39: "Pennsylvania", 40: "Rhode Island", 41: "South Carolina", 42: "South Dakota",
    43: "Tennessee", 44: "Texas", 45: "Utah", 46: "Vermont", 47: "Virginia",
    48: "Washington", 49: "West Virginia", 50: "Wisconsin", 51: "Wyoming", 52: "Other"
}

RACE_MAPPINGS = {
    1: "White",
    2: "Black",
    3: "Other",
    5: "Hispanic"
}

CHRONIC_CONDITIONS = [
    ("ChronicCond_Alzheimer", "Alzheimer's Disease"),
    ("ChronicCond_Heartfailure", "Heart Failure"),
    ("ChronicCond_KidneyDisease", "Kidney Disease"),
    ("ChronicCond_Cancer", "Cancer"),
    ("ChronicCond_ObstrPulmonary", "COPD / Pulmonary"),
    ("ChronicCond_Depression", "Depression"),
    ("ChronicCond_Diabetes", "Diabetes"),
    ("ChronicCond_IschemicHeart", "Ischemic Heart"),
    ("ChronicCond_Osteoporasis", "Osteoporosis"),
    ("ChronicCond_rheumatoidarthritis", "Rheumatoid Arthritis"),
    ("ChronicCond_stroke", "Stroke")
]


def convert_currency(
    value: float,
    source_currency: str = "USD",
    target_currency: str = "INR",
) -> Optional[float]:
    """Converts between the source and display currencies without changing source data."""
    if value is None or pd.isna(value):
        return None
    amount = float(value)
    source = source_currency.upper()
    target = target_currency.upper()
    if source == target:
        return amount
    if source == "USD" and target == "INR":
        return amount * USD_TO_INR_RATE
    if source == "INR" and target == "USD":
        return amount / USD_TO_INR_RATE
    raise ValueError(f"Unsupported currency conversion: {source_currency} to {target_currency}.")


def usd_to_inr(value: float) -> Optional[float]:
    """Converts an unchanged source USD amount to INR for presentation."""
    return convert_currency(value, "USD", "INR")


def format_inr_amount(value: float) -> str:
    """Formats an already-converted INR amount using Indian digit grouping."""
    if value is None or pd.isna(value):
        return "Not available"
    amount = float(value)
    sign = "-" if amount < 0 else ""
    whole, fraction = f"{abs(amount):.2f}".split(".")
    if len(whole) > 3:
        last_three = whole[-3:]
        leading = whole[:-3]
        groups = []
        while len(leading) > 2:
            groups.insert(0, leading[-2:])
            leading = leading[:-2]
        if leading:
            groups.insert(0, leading)
        whole = ",".join(groups + [last_three])
    return f"{sign}₹{whole}.{fraction}"


def format_inr(value: float) -> str:
    """Formats a source USD amount as INR using Indian digit grouping."""
    return format_inr_amount(usd_to_inr(value))


def format_currency(value: float) -> str:
    """Formats source USD as converted INR; use format_usd for source traceability."""
    return format_inr(value)


def format_usd(value: float) -> str:
    """Formats an unconverted source amount for traceability."""
    if value is None or pd.isna(value):
        return "Not available"
    return f"USD {float(value):,.2f}"


def currency_configuration() -> Dict[str, Any]:
    """Returns the configured display conversion and its explicit provenance."""
    return {
        "source_currency": "USD",
        "display_currency": "INR",
        "rate": USD_TO_INR_RATE,
        "effective_date": USD_TO_INR_EFFECTIVE_DATE,
        "illustrative": "illustrative" in USD_TO_INR_EFFECTIVE_DATE.lower(),
    }


def currency_context_text() -> str:
    """Describes the configured display conversion without implying a current quote."""
    config = currency_configuration()
    provenance = (
        "illustrative; not a current market quote"
        if config["illustrative"]
        else "configured; not independently verified as current"
    )
    effective_date = str(config["effective_date"])
    if config["illustrative"] and " (" in effective_date:
        effective_date = effective_date.split(" (", maxsplit=1)[0]
    one_usd_inr = usd_to_inr(1)
    return (
        f"Source monetary values are USD. INR displays use 1 USD = "
        f"₹{one_usd_inr:,.2f} (setting: {effective_date}; {provenance}). "
        "Original USD amounts are preserved."
    )


def format_money_trace(value: float) -> str:
    """Shows converted INR together with the original USD source amount."""
    if value is None or pd.isna(value):
        return "Not available"
    return f"{format_inr(value)} (source: {format_usd(value)})"


def format_date(value) -> str:
    """Formats datetime or string into YYYY-MM-DD format."""
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        dt = pd.to_datetime(value)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def extract_active_conditions(row: pd.Series) -> List[str]:
    """
    Extracts the list of active chronic conditions for a beneficiary record.
    In CMS SynPUF, 1 indicates Yes, 2 indicates No.
    """
    active = []
    for col_name, display_name in CHRONIC_CONDITIONS:
        if col_name in row and row[col_name] == 1:
            active.append(display_name)
    return active


def extract_codes_list(row: pd.Series, prefix: str, max_count: int) -> List[str]:
    """Extracts non-null diagnosis or procedure codes from numbered columns."""
    codes = []
    for i in range(1, max_count + 1):
        col = f"{prefix}_{i}"
        if col not in row or pd.isna(row[col]):
            continue
        value = row[col]
        if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
            codes.append(str(int(value)))
        elif str(value).strip() not in ["", "nan", "None"]:
            codes.append(str(value).strip())
    return codes
