#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
North Conway Weather Revenue Analysis
Created on Sat Apr 12 18:59:35 2025
@author: Vijay Chandra Atheli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

print("NORTH CONWAY WEATHER REVENUE ANALYSIS")
print("=" * 50)

# Load the dataset
df = pd.read_excel('Transactions_Weather_Merged (1).xlsx')
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Data overview
print("\nDATASET OVERVIEW")
print("-" * 30)
print(df.describe())

# Check for missing values 
missing_values = df.isnull().sum()
print(f"\nMissing Values:\n{missing_values[missing_values > 0]}")

# Drop unnecessary columns
columns_to_drop = ['Transaction ID', 'Opened', 'Discount Amount', 'Amount', 'Tax', 'Tip', 
                   'Total Charge', 'Avg_Transaction_Value', 'Tip_Percent']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Create hourly dataset
df_hourly = df.drop_duplicates(subset=['Date', 'Hour']).reset_index(drop=True)
print(f"Hourly dataset created: {df_hourly.shape[0]} unique hour records")

# Create weather condition flags
df_hourly['Is_Snow'] = (
    (df_hourly['precip_mm'] > 0) &
    (df_hourly['temp_C'] <= 2) & 
    (df_hourly['visibility_km'] <= 5)
).astype(int)

df_hourly['Is_Sunny'] = (
    (df_hourly['precip_mm'] == 0) &
    (df_hourly['cloudcover'] <= 30) &
    (df_hourly['visibility_km'] >= 8)
).astype(int)

weather_counts = {
    'Rainy Hours': df_hourly['Is_Rainy'].sum(),
    'Snowy Hours': df_hourly['Is_Snow'].sum(),
    'Sunny Hours': df_hourly['Is_Sunny'].sum(),
    'Other Weather': len(df_hourly) - df_hourly['Is_Rainy'].sum() - df_hourly['Is_Snow'].sum() - df_hourly['Is_Sunny'].sum()
}

for condition, count in weather_counts.items():
    print(f"{condition}: {count} ({count/len(df_hourly)*100:.1f}%)")

df_hourly.to_excel('df_hourly_north_conway.xlsx', index=False)
print(f"\nProcessed data saved to 'df_hourly_north_conway.xlsx'")

# Create visualizations
print("\nCreating visualizations...")

plt.figure(figsize=(15, 12))

# Weather Distribution
plt.subplot(3, 2, 1)
weather_data = list(weather_counts.values())
weather_labels = list(weather_counts.keys())
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
plt.pie(weather_data, labels=weather_labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Weather Condition Distribution', fontsize=14, fontweight='bold')

# Revenue by Weather Condition
plt.subplot(3, 2, 2)
weather_revenue = {
    'Rainy': df_hourly[df_hourly['Is_Rainy']==1]['Total_Hourly_Revenue'].mean(),
    'Snowy': df_hourly[df_hourly['Is_Snow']==1]['Total_Hourly_Revenue'].mean(),
    'Sunny': df_hourly[df_hourly['Is_Sunny']==1]['Total_Hourly_Revenue'].mean(),
    'Other': df_hourly[(df_hourly['Is_Rainy']==0) & (df_hourly['Is_Snow']==0) & (df_hourly['Is_Sunny']==0)]['Total_Hourly_Revenue'].mean()
}
bars = plt.bar(weather_revenue.keys(), weather_revenue.values(), color=colors)
plt.title('Average Revenue by Weather Condition', fontsize=14, fontweight='bold')
plt.ylabel('Average Hourly Revenue ($)')
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'${height:.0f}', ha='center', va='bottom', fontweight='bold')

# Revenue Distribution
plt.subplot(3, 2, 3)
plt.hist(df_hourly['Total_Hourly_Revenue'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Revenue Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Hourly Revenue ($)')
plt.ylabel('Frequency')
plt.axvline(df_hourly['Total_Hourly_Revenue'].mean(), color='red', linestyle='--', 
            label=f'Mean: ${df_hourly["Total_Hourly_Revenue"].mean():.0f}')
plt.legend()

# Temperature vs Revenue Scatter
plt.subplot(3, 2, 4)
scatter = plt.scatter(df_hourly['temp_C'], df_hourly['Total_Hourly_Revenue'], 
                     alpha=0.6, c=df_hourly['precip_mm'], cmap='viridis', s=30)
plt.colorbar(scatter, label='Precipitation (mm)')
plt.title('Temperature vs Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Temperature (°C)')
plt.ylabel('Hourly Revenue ($)')

# Hourly Revenue Pattern
plt.subplot(3, 2, 5)
hourly_avg = df_hourly.groupby('Hour')['Total_Hourly_Revenue'].mean()
plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=6)
plt.title('Average Revenue by Hour of Day', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day')
plt.ylabel('Average Revenue ($)')
plt.xticks(range(0, 24, 2))
plt.grid(True, alpha=0.3)

# Weather Impact Box Plot
plt.subplot(3, 2, 6)
weather_conditions = ['Rainy', 'Snowy', 'Sunny', 'Other']
revenue_data = [
    df_hourly[df_hourly['Is_Rainy']==1]['Total_Hourly_Revenue'].values,
    df_hourly[df_hourly['Is_Snow']==1]['Total_Hourly_Revenue'].values,
    df_hourly[df_hourly['Is_Sunny']==1]['Total_Hourly_Revenue'].values,
    df_hourly[(df_hourly['Is_Rainy']==0) & (df_hourly['Is_Snow']==0) & (df_hourly['Is_Sunny']==0)]['Total_Hourly_Revenue'].values
]
plt.boxplot(revenue_data, labels=weather_conditions)
plt.title('Revenue Distribution by Weather', fontsize=14, fontweight='bold')
plt.ylabel('Hourly Revenue ($)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Build predictive model
print("\nBuilding predictive model...")

# Add lagged variables
df_hourly['Revenue_Lag1'] = df_hourly['Total_Hourly_Revenue'].shift(1)
df_hourly['Revenue_Lag2'] = df_hourly['Total_Hourly_Revenue'].shift(2)
df_hourly = df_hourly.dropna()

print(f"Dataset after adding lags: {df_hourly.shape[0]} rows")

# Define features and target
X = df_hourly[['Revenue_Lag1', 'Revenue_Lag2', 'Is_Rainy', 'Is_Snow', 'Is_Sunny']]
y = df_hourly['Total_Hourly_Revenue']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train model
lr_model = LinearRegression().fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\nMODEL PERFORMANCE")
print("-" * 25)
print("Linear Regression Coefficients:")
for feature, coef in zip(X.columns, lr_model.coef_):
    print(f"{feature:15}: {coef:8.3f}")

print(f"\nIntercept: {lr_model.intercept_:.3f}")
print(f"\nModel Performance Metrics:")
print(f"   R² Score: {r2:.3f} ({r2*100:.1f}% variance explained)")
print(f"   MAE:      ${mae:.2f}")
print(f"   RMSE:     ${rmse:.2f}")

# Model visualization
plt.figure(figsize=(15, 5))

# Actual vs Predicted
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Revenue ($)')
plt.ylabel('Predicted Revenue ($)')
plt.title(f'Actual vs Predicted\nR² = {r2:.3f}', fontweight='bold')
plt.grid(True, alpha=0.3)

# Residuals
plt.subplot(1, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, color='orange')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Revenue ($)')
plt.ylabel('Residuals ($)')
plt.title('Residual Plot', fontweight='bold')
plt.grid(True, alpha=0.3)

# Feature Importance
plt.subplot(1, 3, 3)
feature_importance = abs(lr_model.coef_)
features = X.columns
bars = plt.barh(features, feature_importance, color='lightcoral')
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance', fontweight='bold')
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print(f"\nANALYSIS SUMMARY")
print("=" * 50)
print(f"Business: North Conway Location")
print(f"Data Period: {df_hourly['Date'].min()} to {df_hourly['Date'].max()}")
print(f"Total Hours Analyzed: {len(df_hourly)}")
print(f"Average Hourly Revenue: ${df_hourly['Total_Hourly_Revenue'].mean():.2f}")
print(f"Weather Impact:")
for condition, avg_revenue in weather_revenue.items():
    print(f"   {condition}: ${avg_revenue:.2f}")
print(f"Model Accuracy: {r2*100:.1f}% of revenue variance explained")
print(f"Prediction Error: ±${rmse:.2f} on average")
print(f"\nAnalysis Complete! Data saved to 'df_hourly_north_conway.xlsx'")