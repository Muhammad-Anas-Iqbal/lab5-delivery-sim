import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. CREATE DELIVERY TIMES DATASET
# --------------------------------------------------

np.random.seed(42)

# Generate 100 synthetic delivery times
delivery_times = np.random.normal(
    loc=32,
    scale=8,
    size=100
)

# Add some unusually long delivery times
delivery_times[0] = 68
delivery_times[1] = 71
delivery_times[2] = 75


data = {
    "OrderID": range(1, 101),
    "DeliveryTime(min)": delivery_times
}


# Save dataset as CSV
pd.DataFrame(data).to_csv(
    "delivery_times.csv",
    index=False
)


# --------------------------------------------------
# 2. LOAD THE CSV FILE
# --------------------------------------------------

df = pd.read_csv("delivery_times.csv")

print("First 5 rows:")
print(df.head())


# --------------------------------------------------
# 3. CALCULATE MEAN AND STANDARD DEVIATION
# --------------------------------------------------

mean_time = df["DeliveryTime(min)"].mean()

std_time = df["DeliveryTime(min)"].std()

print("\nReal delivery times")
print("Mean:", mean_time, "min")
print("Standard Deviation:", std_time, "min")


# --------------------------------------------------
# 4. SIMULATE EXPECTED DELIVERY TIMES
# --------------------------------------------------

simulated_times = np.random.normal(
    loc=mean_time,
    scale=std_time,
    size=100
)

print(
    "\nSimulated 100 values from Normal distribution"
)

print("Mean =", mean_time)
print("Standard Deviation =", std_time)


# --------------------------------------------------
# 5. OVERLAPPING HISTOGRAMS
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["DeliveryTime(min)"],
    bins=10,
    alpha=0.5,
    label="Real Delivery Times"
)

plt.hist(
    simulated_times,
    bins=10,
    alpha=0.5,
    label="Simulated Delivery Times"
)

plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Frequency")
plt.title("Real vs Simulated Delivery Times")

plt.legend()

plt.tight_layout()

plt.savefig(
    "delivery_comparison_histogram.png"
)

print(
    "\nSaved delivery_comparison_histogram.png"
)


# --------------------------------------------------
# 6. CHECK MISSING VALUES
# --------------------------------------------------

print("\nMissing values:")
print(
    df["DeliveryTime(min)"].isnull().sum()
)


# --------------------------------------------------
# 7. BOXPLOT
# --------------------------------------------------

plt.figure(figsize=(5, 4))

plt.boxplot(
    df["DeliveryTime(min)"]
)

plt.ylabel("Delivery Time (minutes)")
plt.title("Delivery Time Boxplot")

plt.tight_layout()

plt.savefig(
    "delivery_boxplot.png"
)
plt.show()
print("Saved delivery_boxplot.png")


# --------------------------------------------------
# 8. CALCULATE PERCENTILES
# --------------------------------------------------

Q1 = df["DeliveryTime(min)"].quantile(0.25)

Q2 = df["DeliveryTime(min)"].quantile(0.50)

Q3 = df["DeliveryTime(min)"].quantile(0.75)

print("\nPercentiles:")

print("25th percentile:", Q1)
print("50th percentile:", Q2)
print("75th percentile:", Q3)


# --------------------------------------------------
# 9. CALCULATE IQR
# --------------------------------------------------

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR

print("\nIQR:", IQR)

print(
    "IQR Outlier Bounds:",
    lower_bound,
    "-",
    upper_bound
)


# --------------------------------------------------
# 10. FIND OUTLIERS
# --------------------------------------------------

outliers = df[
    (df["DeliveryTime(min)"] < lower_bound)
    |
    (df["DeliveryTime(min)"] > upper_bound)
]

print("\nOutliers detected:")
print(outliers)


# Print only delivery times of outliers
print("\nOutlier delivery times:")

print(
    outliers["DeliveryTime(min)"].tolist()
)


# --------------------------------------------------
# 11. CALCULATE OUTLIER PERCENTAGE
# --------------------------------------------------

outlier_count = len(outliers)

percentage = (
    outlier_count / len(df)
) * 100

print(
    "\nNumber of outliers:",
    outlier_count
)

print(
    "Percentage flagged as outliers:",
    percentage,
    "%"
)


# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------

print(
    "\nSaved delivery_comparison_histogram.png "
    "and delivery_boxplot.png"
)