import pandas as pd

RENAME_MAP = {
    "fuel_level": "fuel",
    "ros_main__generator_controller__hatz_info__fuel_level__value": "fuel",
    "ros_main__generator_controller__hatz_info__engine_speed__value": "engine_speed",
    "ros_main__generator_controller__hatz_info__charger_output_current__value": "charger_current",
    "ros_battery_state__voltage_V": "battery_voltage",
    "ros_main__inverse_kinematics__cmd__data__speed": "cmd_speed",
}

def read_standardize(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns=RENAME_MAP).copy()
    if "datetime" not in df.columns:
        raise KeyError("Expected 'datetime' column.")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def make_work(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["datetime", "fuel", "cmd_speed"] if c in df.columns]
    if "datetime" not in cols or "fuel" not in cols:
        raise KeyError("Need 'datetime' and 'fuel'.")
    return df[cols].copy()