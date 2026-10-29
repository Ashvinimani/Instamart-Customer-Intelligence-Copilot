import os
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_customers():
    df = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))

    df["city"] = df["city"].astype("string").str.strip().str.title()

    df["gender"] = (
        df["gender"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({
            "m": "male",
            "f": "female",
            "": pd.NA
        })
    )

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[~df["age"].between(18, 100), "age"] = pd.NA

    df["age"] = df["age"].fillna(df["age"].median())

    df.to_csv(
        os.path.join(PROCESSED_DIR, "customers_clean.csv"),
        index=False
    )
    print("Customers cleaned:", df.shape)


def clean_orders():
    df = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))

    df = df.drop_duplicates()

    for column in [
        "quantity",
        "discount_pct",
        "order_amount",
        "delivery_minutes"
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df.loc[df["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

    df["order_amount"] = df["order_amount"].fillna(
        df["order_amount"].median()
    )

    df["delivery_minutes"] = df["delivery_minutes"].fillna(
        df["delivery_minutes"].median()
    )

    df.to_csv(
        os.path.join(PROCESSED_DIR, "orders_clean.csv"),
        index=False
    )
    print("Orders cleaned:", df.shape)


def clean_products():
    df = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))

    df = df.drop_duplicates()
    df["product_name"] = df["product_name"].astype("string").str.strip()
    df["category"] = df["category"].astype("string").str.strip().str.title()
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")

    df.to_csv(
        os.path.join(PROCESSED_DIR, "products_clean.csv"),
        index=False
    )
    print("Products cleaned:", df.shape)


def clean_tickets():
    df = pd.read_csv(os.path.join(RAW_DIR, "tickets.csv"))

    df = df.drop_duplicates()
    df["text"] = df["text"].fillna("").astype("string").str.strip()
    df = df[df["text"].str.len() > 0].copy()

    df.to_csv(
        os.path.join(PROCESSED_DIR, "tickets_clean.csv"),
        index=False
    )
    print("Tickets cleaned:", df.shape)


if __name__ == "__main__":
    clean_customers()
    clean_orders()
    clean_products()
    clean_tickets()

    print("Data cleaning completed successfully!")
