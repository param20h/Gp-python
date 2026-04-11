import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, norm
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import classification_report


sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# DATA MANIPULATION WITH NumPy & Pandas

print("=" * 65)
print("  SECTION 1: DATA LOADING & CLEANING")
print("=" * 65)

# ---------- Load Dataset ----------
df = pd.read_csv("global_warming_dataset.csv")
print(f"\n Dataset loaded successfully.")
print(f"       Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ----------Inspect Dataset ----------
print("--- First 5 Rows ---")
print(df.head())

print("\n--- Column Data Types ---")
print(df.dtypes)


# ---------- Handle Missing Values ----------
print("\n--- Missing Values Per Column ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "  No missing values found.")


# ---------- After missing values check — ADD THIS ----------
co2_arr  = np.array(df['CO2_Emissions'])
co2_norm = (co2_arr - co2_arr.min()) / (co2_arr.max() - co2_arr.min())
df['CO2_Normalized']    = co2_norm
df['Decade']            = (df['Year'] // 10) * 10
df['Era']               = np.where(df['Year'] < 1980, 'Pre-1980', 'Post-1980')
df['Emission_Category'] = pd.cut(df['CO2_Normalized'],
                                  bins=[-np.inf, 0.33, 0.66, np.inf],
                                  labels=['Low', 'Medium', 'High'])


print("\n Data cleaning and preparation complete.\n")
