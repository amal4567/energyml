from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("data/cleaned_dataset.csv")

print("="*60)
print("DATASET OVERVIEW")
print("="*60)

print("Shape :", df.shape)

print("\nColumns :")
print(df.columns.tolist())

print("\nInfo :")
print(df.info())

print("\nMissing Values :")
print(df.isnull().sum())

# =====================================================
# DESCRIPTIVE STATISTICS
# =====================================================

print("\nDescriptive Statistics")
print(df.describe())

# =====================================================
# TARGET DISTRIBUTION
# =====================================================

print("Nombre de lignes :", len(df))  # devrait afficher 15000

plt.figure(figsize=(10, 5))
sns.histplot(
    df["meter_reading"],
    bins=100,
    kde=True
)
plt.title(f"Distribution of Meter Reading (n={len(df)})")
plt.xlabel("Meter Reading")
plt.ylabel("Frequency")
plt.ylim(0, 15000)
plt.show()


# =====================================================
# LOG TRANSFORMATION
# =====================================================

df["log_meter_reading"] = np.log1p(
    df["meter_reading"]
)

plt.figure(figsize=(10, 5))

sns.histplot(
    df["log_meter_reading"],
    bins=100,
    kde=True
)

plt.title("Log Transformed Meter Reading")

plt.show()

# =====================================================
# BUILDING TYPE DISTRIBUTION
# =====================================================

plt.figure(figsize=(12, 6))

sns.countplot(
    y=df["primary_use"],
    order=df["primary_use"].value_counts().index
)

plt.title("Building Type Distribution")

plt.show()

# =====================================================
# ENERGY CONSUMPTION BY BUILDING TYPE
# =====================================================

plt.figure(figsize=(14, 6))

sns.boxplot(
    data=df,
    x="primary_use",
    y="log_meter_reading"
)

plt.xticks(rotation=90)

plt.title("Energy Consumption by Building Type")

plt.show()

# =====================================================
# ENERGY CONSUMPTION BY METER TYPE
# =====================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="meter",
    y="log_meter_reading"
)

plt.title("Consumption by Meter Type")

plt.show()

# =====================================================
# SQUARE FEET VS CONSUMPTION
# =====================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="square_feet",
    y="log_meter_reading"
)

plt.title("Building Area vs Consumption")

plt.show()

# =====================================================
# AIR TEMPERATURE VS CONSUMPTION
# =====================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="air_temperature",
    y="log_meter_reading"
)

plt.title("Temperature vs Consumption")

plt.show()

# =====================================================
# WEATHER FEATURES DISTRIBUTION
# =====================================================

weather_features = [

    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed"
]

for col in weather_features:

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        x=df[col]
    )

    plt.title(f"Boxplot - {col}")

    plt.show()

# =====================================================
# TIMESTAMP ANALYSIS
# =====================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df["month"] = (
    df["timestamp"].dt.month
)

df["hour"] = (
    df["timestamp"].dt.hour
)

df["dayofweek"] = (
    df["timestamp"].dt.dayofweek
)

# =====================================================
# MONTHLY CONSUMPTION
# =====================================================

monthly = df.groupby(
    "month"
)["log_meter_reading"].mean()

plt.figure(figsize=(10, 5))

monthly.plot(
    marker="o"
)

plt.title("Average Consumption by Month")

plt.ylabel("Average Consumption")

plt.grid()

plt.show()

# =====================================================
# HOURLY CONSUMPTION
# =====================================================

hourly = df.groupby(
    "hour"
)["log_meter_reading"].mean()

plt.figure(figsize=(10, 5))

hourly.plot(
    marker="o"
)

plt.title("Average Consumption by Hour")

plt.ylabel("Average Consumption")

plt.grid()

plt.show()

# =====================================================
# DAY OF WEEK CONSUMPTION
# =====================================================

daily = df.groupby(
    "dayofweek"
)["log_meter_reading"].mean()

plt.figure(figsize=(8, 5))

daily.plot(
    marker="o"
)

plt.title("Average Consumption by Day of Week")

plt.grid()

plt.show()

# =====================================================
# CORRELATION MATRIX
# =====================================================

numeric_df = df.select_dtypes(
    include=np.number
)

corr = numeric_df.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.show()

# =====================================================
# TOP CORRELATIONS WITH TARGET
# =====================================================

print("\nTop Correlations with Meter Reading")

target_corr = corr[
    "meter_reading"
].sort_values(
    ascending=False
)

print(target_corr)

# =====================================================
# OUTLIER DETECTION
# =====================================================

Q1 = df["meter_reading"].quantile(0.25)

Q3 = df["meter_reading"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR

outliers = df[
    (df["meter_reading"] < lower)
    |
    (df["meter_reading"] > upper)
]

print("\nOutliers detected :", len(outliers))

print(
    "Percentage :",
    round(
        len(outliers)/len(df)*100,
        2
    ),
    "%"
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================


eda_df = df.copy()

eda_df = pd.get_dummies(
    eda_df,
    columns=["primary_use"],
    drop_first=True
)

eda_df = eda_df.drop(
    columns=["timestamp"]
)

X = eda_df.drop(
    columns=[
        "meter_reading",
        "log_meter_reading"
    ]
)

y = eda_df["meter_reading"]

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        rf.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features")

print(
    importance.head(20)
)

plt.figure(figsize=(10, 8))

sns.barplot(
    data=importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 15 Important Features"
)

plt.show()

print("="*60)
print("EDA FINISHED SUCCESSFULLY")
print("="*60)
