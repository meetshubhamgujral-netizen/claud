"""Generate a synthetic customer-churn dataset for demoing the dashboard."""
import numpy as np, pandas as pd

def make(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 25, n).clip(15, 150).round(2)
    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:06d}" for i in range(n)],
        "age": rng.integers(18, 85, n),
        "gender": rng.choice(["Male", "Female"], n),
        "region": rng.choice(["North", "South", "East", "West", "Central"], n),
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": (monthly * tenure * rng.uniform(0.9, 1.1, n)).round(2),
        "contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55,.25,.20]),
        "support_calls": rng.poisson(2, n),
        "satisfaction": rng.integers(1, 6, n),
        "signup_date": pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 1400, n), unit="D"),
    })
    # churn correlated with tenure, contract, satisfaction, support calls
    logit = (-0.04*df.tenure_months + 0.5*(df.contract=="Month-to-month")
             - 0.4*df.satisfaction + 0.25*df.support_calls + rng.normal(0,1,n) + 1.0)
    df["churn"] = (1/(1+np.exp(-logit)) > 0.5).astype(int)
    # inject some missing values for realism
    df.loc[df.sample(frac=0.04, random_state=seed).index, "total_charges"] = np.nan
    return df

if __name__ == "__main__":
    make().to_csv("sample_customer_churn.csv", index=False)
    print("Wrote sample_customer_churn.csv")
