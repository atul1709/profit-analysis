import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load Dataset
df = pd.read_csv("business_data.csv")
print("Dataset Loaded Successfully\n")
print(df.head())

# Handle Missing Values
# Fill only numeric columns with mean
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Convert Month to category (optional)
df['Month'] = df['Month'].astype('category')

print("\nAfter Cleaning:\n")
print(df.head())


# Feature Engineering: Profit
df["Profit"] = df["Revenue"] - df["Cost"] - df["Marketing_Spend"] - df["Operational_Expense"]

print("\nProfit column added:\n")
print(df[["Month", "Revenue", "Cost", "Marketing_Spend", "Operational_Expense", "Profit"]])

# EDA Visualizations

# Profit Trend
plt.figure(figsize=(10, 5))
plt.plot(df['Month'].astype(str), df['Profit'], marker='o')
plt.title("Monthly Profit Trend")
plt.xlabel("Month")
plt.ylabel("Profit")
plt.grid(True)
plt.tight_layout()
plt.show()

# Revenue vs Cost
plt.figure(figsize=(10, 5))
x = np.arange(len(df['Month']))
width = 0.35

plt.bar(x - width/2, df['Revenue'], width, label='Revenue')
plt.bar(x + width/2, df['Cost'], width, label='Cost')

plt.xticks(x, df['Month'].astype(str), rotation=45)
plt.xlabel("Month")
plt.ylabel("Amount")
plt.title("Revenue vs Cost by Month")
plt.legend()
plt.tight_layout()
plt.show()

# Correlation Heatmap
corr = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(8, 6))
im = plt.imshow(corr, interpolation='nearest')
plt.title("Correlation Heatmap")
plt.colorbar(im, fraction=0.046, pad=0.04)

# Labels
ticks = np.arange(len(corr.columns))
plt.xticks(ticks, corr.columns, rotation=45, ha='right')
plt.yticks(ticks, corr.columns)

# Annotate cells
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=8)

plt.tight_layout()
plt.show()

# Key Profit Drivers
profit_corr = corr["Profit"].sort_values(ascending=False)
print("\nCorrelation of Each Feature with Profit:\n")
print(profit_corr)

# Final Insights
print("\n==== BUSINESS INSIGHTS ====\n")
print("1. Revenue has the highest positive impact on Profit.")
print("2. Cost and Operational Expense have strong negative correlations (they reduce profit).")
print("3. Marketing Spend needs analysis; high spend may not always increase profit.")
print("4. Profit varies across months — possible seasonality effect.")
print("5. Units Sold also positively affects revenue and profit.\n")

print("\nAnalysis Completed Successfully!")
