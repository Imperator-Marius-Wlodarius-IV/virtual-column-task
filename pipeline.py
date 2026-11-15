from pathlib import Path
import pandas as pd
from solution import add_virtual_column


def clean_sales_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # Normalize column headers to lowercase snake_case (letters and underscores only)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Remove repeated header rows and summary TOTAL rows
    order_id_clean = df["order_id"].astype(str).str.strip()
    df = df[
        (order_id_clean.str.lower() != "order id")
        & (order_id_clean.str.upper() != "TOTAL")
    ].copy()

    # Remove exact duplicate rows
    df = df.drop_duplicates().copy()

    # Normalize text fields (whitespace & consistent title casing for region)
    df["order_id"] = df["order_id"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip().str.title()
    df["product"] = df["product"].astype(str).str.strip()
    df["order_date"] = pd.to_datetime(
        df["order_date"].astype(str).str.strip(), format="%Y-%m-%d"
    )

    # Clean quantity: remove internal thousands-separator spaces and parse as integer
    df["quantity"] = (
        df["quantity"]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .astype(int)
    )

    # Clean unit_price: normalize comma decimal separator and impute missing prices per product
    unit_price_str = (
        df["unit_price"]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    df["unit_price"] = pd.to_numeric(unit_price_str, errors="coerce")
    df["unit_price"] = df["unit_price"].fillna(
        df.groupby("product")["unit_price"].transform("median")
    )

    # Clean discount: normalize decimal separator and fill empty discounts with 0.0
    discount_str = (
        df["discount"]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    df["discount"] = pd.to_numeric(discount_str, errors="coerce").fillna(0.0)

    return df


def compute_revenue_by_region(df: pd.DataFrame) -> pd.DataFrame:
    enriched_df = add_virtual_column(
        df, "quantity * unit_price - discount", "revenue"
    )
    if enriched_df.empty:
        raise ValueError("Failed to compute virtual column 'revenue'.")

    summary = (
        enriched_df.groupby("region", as_index=False)["revenue"]
        .sum()
        .sort_values("region")
        .reset_index(drop=True)
    )
    summary["revenue"] = summary["revenue"].round(2)
    return summary


def run_pipeline(
    input_path: str | Path = "sales.csv",
    output_path: str | Path = "revenue_by_region.csv",
) -> pd.DataFrame:
    raw_df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    cleaned_df = clean_sales_data(raw_df)
    revenue_df = compute_revenue_by_region(cleaned_df)
    revenue_df.to_csv(output_path, index=False)
    return revenue_df


if __name__ == "__main__":
    run_pipeline()