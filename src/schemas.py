"""Input data contracts. Every uploaded CSV is parsed and checked here before the pipeline sees it."""
import pandas as pd

SCHEMAS = {
    "daily": {
        "label": "Daily consumption",
        "filename": "daily_consumption.csv",
        "columns": {
            "meter_id": "str",
            "reading_date": "date",
            "kwh_import": "float",
            "intervals_expected": "int",
            "intervals_received": "int",
            "voltage_present_pct": "float",
        },
        "key": ["meter_id", "reading_date"],
    },
    "master": {
        "label": "Meter master",
        "filename": "meter_master.csv",
        "columns": {
            "meter_id": "str",
            "consumer_id": "str",
            "dt_id": "str",
            "subdivision_id": "str",
            "tariff_category": "str",
            "sanctioned_load_kw": "float",
            "phase_type": "str",
            "status": "str",
            "status_changed_on": "date",
            "meter_installed_on": "date",
            "has_registered_der": "bool",
        },
        "key": ["meter_id"],
    },
    "officers": {
        "label": "SE directory",
        "filename": "officers.csv",
        "columns": {
            "subdivision_id": "str",
            "se_name": "str",
            "se_email": "str",
        },
        "key": ["subdivision_id"],
    },
}

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f", ""}


def template(kind: str) -> pd.DataFrame:
    """An empty dataframe with the required header, for the template download."""
    return pd.DataFrame(columns=list(SCHEMAS[kind]["columns"]))


def parse(raw: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, list[str]]:
    """Coerce a raw CSV (read with dtype=str) to the contract. Returns (df, errors)."""
    spec = SCHEMAS[kind]
    errors: list[str] = []
    raw = raw.rename(columns=lambda c: str(c).strip())

    missing = [c for c in spec["columns"] if c not in raw.columns]
    if missing:
        return raw, [f"Missing columns: {', '.join(missing)}"]
    if raw.empty:
        return raw, ["File has no rows"]

    df = raw[list(spec["columns"])].copy()
    for col, typ in spec["columns"].items():
        s = df[col].astype("string").str.strip()
        if typ == "str":
            df[col] = s
            if s.isna().any() or (s == "").any():
                errors.append(f"{col}: {int((s.isna() | (s == '')).sum())} blank values")
        elif typ == "date":
            parsed = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")
            bad = parsed.isna() & s.notna() & (s != "")
            if bad.any():
                errors.append(f"{col}: {int(bad.sum())} values not in YYYY-MM-DD format (e.g. '{s[bad].iloc[0]}')")
            df[col] = parsed
        elif typ in ("float", "int"):
            parsed = pd.to_numeric(s, errors="coerce")
            bad = parsed.isna() & s.notna() & (s != "")
            if bad.any():
                errors.append(f"{col}: {int(bad.sum())} non-numeric values (e.g. '{s[bad].iloc[0]}')")
            df[col] = parsed
        elif typ == "bool":
            low = s.str.lower().fillna("")
            bad = ~low.isin(_TRUE | _FALSE)
            if bad.any():
                errors.append(f"{col}: {int(bad.sum())} values not true/false (e.g. '{s[bad].iloc[0]}')")
            df[col] = low.isin(_TRUE)

    dupes = df.duplicated(spec["key"]).sum()
    if dupes:
        errors.append(f"{int(dupes)} duplicate rows on {' + '.join(spec['key'])}")

    if kind == "daily":
        if (df["kwh_import"] < 0).any():
            errors.append("kwh_import: negative values present")
        if (df["intervals_received"] > df["intervals_expected"]).any():
            errors.append("intervals_received exceeds intervals_expected on some rows")
    if kind == "master":
        if df["meter_installed_on"].isna().any():
            errors.append("meter_installed_on: blank values")
    if kind == "officers":
        bad_email = ~df["se_email"].str.contains(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", regex=True, na=False)
        if bad_email.any():
            errors.append(f"se_email: {int(bad_email.sum())} invalid addresses")

    return df, errors
