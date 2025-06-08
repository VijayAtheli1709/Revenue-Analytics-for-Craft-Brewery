#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boston Weather Revenue Analysis with Comprehensive Visualizations
Created on Sun Apr 13 18:47:53 2025
@author: Enhanced Analysis
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("Set2")

print("BOSTON WEATHER REVENUE ANALYSIS")
print("=" * 50)

# Load transaction data
df = pd.read_excel('Transactions_Weather_Merged (1).xlsx')
print(f"Transaction data loaded: {df.shape}")

# Drop old weather columns
df = df.drop(columns=['precip_mm', 'temp_C', 'visibility_km', 'cloudcover'], errors='ignore')

# Load Boston weather data
print("Loading Boston weather data...")
weather = pd.read_excel('Boston_hourly_weather.xlsx')

weather['datetime'] = pd.to_datetime(weather['date'] + ' ' + weather['time'])
weather['Date'] = pd.to_datetime(weather['datetime'].dt.date)
weather['Hour'] = weather['datetime'].dt.hour
weather = weather.drop(columns=['date', 'time', 'datetime'])

print(f"Boston weather data loaded: {weather.shape}")
print(f"Weather data range: {weather['Date'].min()} to {weather['Date'].max()}")

# Ensure Date column is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Drop unnecessary transaction columns
transaction_cols_to_drop = ['Transaction ID', 'Opened', 'Discount Amount', 'Amount', 'Tax', 'Tip',
                           'Total Charge', 'Avg_Transaction_Value', 'Tip_Percent']
df = df.drop(columns=transaction_cols_to_drop, errors='ignore')

# Create hourly dataset
df_hourly = df.drop_duplicates(subset=['Date', 'Hour']).reset_index(drop=True)
print(f"Hourly data created: {df_hourly.shape[0]} unique hours")

# Merge Boston weather
df_hourly = pd.merge(df_hourly, weather, on=['Date', 'Hour'], how='left')
print(f"After merging with Boston weather: {df_hourly.shape}")

# Check merge success
merged_weather_data = df_hourly[['precip_mm', 'temp_C', 'visibility_km', 'cloudcover']].notna().sum()
print(f"Weather data availability:\n{merged_weather_data}")

# Define weather flags
print("\nCreating weather condition flags...")

df_hourly['Is_Rainy'] = (df_hourly['precip_mm'] > 0).astype(int)

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

# Weather condition summary
weather_summary = {
    'Rainy Hours': df_hourly['Is_Rainy'].sum(),
    'Snowy Hours': df_hourly['Is_Snow'].sum(), 
    'Sunny Hours': df_hourly['Is_Sunny'].sum(),
    'Other Weather': len(df_hourly) - df_hourly['Is_Rainy'].sum() - df_hourly['Is_Snow'].sum() - df_hourly['Is_Sunny'].sum()
}

print("Weather Distribution:")
for condition, count in weather_summary.items():
    print(f"  {condition}: {count} hours ({count/len(df_hourly)*100:.1f}%)")

# Create comprehensive visualizations
print("\nCreating comprehensive visualizations...")

fig = plt.figure(figsize=(20, 16))

# Weather Distribution Pie Chart
plt.subplot(4, 3, 1)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
plt.pie(weather_summary.values(), labels=weather_summary.keys(), autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('Boston Weather Distribution', fontsize=14, fontweight='bold')

# Revenue by Weather Condition
plt.subplot(4, 3, 2)
weather_revenue_data = []
weather_labels = []
for weather_type, flag_col in [('Rainy', 'Is_Rainy'), ('Snowy', 'Is_Snow'), ('Sunny', 'Is_Sunny')]:
    avg_revenue = df_hourly[df_hourly[flag_col]==1]['Total_Hourly_Revenue'].mean()
    weather_revenue_data.append(avg_revenue)
    weather_labels.append(weather_type)

# Add "Other" weather
other_mask = (df_hourly['Is_Rainy']==0) & (df_hourly['Is_Snow']==0) & (df_hourly['Is_Sunny']==0)
other_revenue = df_hourly[other_mask]['Total_Hourly_Revenue'].mean()
weather_revenue_data.append(other_revenue)
weather_labels.append('Other')

bars = plt.bar(weather_labels, weather_revenue_data, color=colors)
plt.title('Average Revenue by Weather (Boston)', fontsize=14, fontweight='bold')
plt.ylabel('Average Hourly Revenue ($)')
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'${height:.0f}', ha='center', va='bottom', fontweight='bold')

# Temperature Distribution
plt.subplot(4, 3, 3)
plt.hist(df_hourly['temp_C'].dropna(), bins=30, alpha=0.7, color='coral', edgecolor='black')
plt.title('Boston Temperature Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.axvline(df_hourly['temp_C'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df_hourly["temp_C"].mean():.1f}°C')
plt.legend()

# Precipitation vs Revenue
plt.subplot(4, 3, 4)
plt.scatter(df_hourly['precip_mm'], df_hourly['Total_Hourly_Revenue'], 
           alpha=0.6, c=df_hourly['temp_C'], cmap='coolwarm', s=30)
plt.colorbar(label='Temperature (°C)')
plt.title('Precipitation vs Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Hourly Revenue ($)')

# Hourly Revenue Pattern
plt.subplot(4, 3, 5)
hourly_avg = df_hourly.groupby('Hour')['Total_Hourly_Revenue'].mean()
hourly_std = df_hourly.groupby('Hour')['Total_Hourly_Revenue'].std()
plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=3, markersize=8, color='steelblue')
plt.fill_between(hourly_avg.index, hourly_avg.values - hourly_std.values, 
                 hourly_avg.values + hourly_std.values, alpha=0.2, color='steelblue')
plt.title('Average Revenue by Hour (Boston)', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day')
plt.ylabel('Average Revenue ($)')
plt.xticks(range(0, 24, 2))
plt.grid(True, alpha=0.3)

# Cloud Cover vs Revenue
plt.subplot(4, 3, 6)
plt.scatter(df_hourly['cloudcover'], df_hourly['Total_Hourly_Revenue'], 
           alpha=0.6, color='lightblue', s=30)
plt.title('Cloud Cover vs Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Cloud Cover (%)')
plt.ylabel('Hourly Revenue ($)')

# Visibility Impact
plt.subplot(4, 3, 7)
df_hourly['Visibility_Bin'] = pd.cut(df_hourly['visibility_km'], 
                                    bins=[0, 5, 10, 15, float('inf')], 
                                    labels=['Poor (<5km)', 'Fair (5-10km)', 'Good (10-15km)', 'Excellent (>15km)'])
visibility_revenue = df_hourly.groupby('Visibility_Bin')['Total_Hourly_Revenue'].mean()
bars = plt.bar(range(len(visibility_revenue)), visibility_revenue.values, 
               color=['red', 'orange', 'yellow', 'green'])
plt.title('Revenue by Visibility', fontsize=14, fontweight='bold')
plt.xlabel('Visibility Range')
plt.ylabel('Average Revenue ($)')
plt.xticks(range(len(visibility_revenue)), visibility_revenue.index, rotation=45)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'${height:.0f}', ha='center', va='bottom', fontweight='bold')

# Weather Correlation Heatmap
plt.subplot(4, 3, 8)
weather_corr_data = df_hourly[['Total_Hourly_Revenue', 'temp_C', 'precip_mm', 'visibility_km', 'cloudcover']].corr()
sns.heatmap(weather_corr_data, annot=True, cmap='RdBu_r', center=0, 
            square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('Weather-Revenue Correlation', fontsize=14, fontweight='bold')

# Revenue Distribution Box Plot
plt.subplot(4, 3, 9)
weather_data_for_box = []
box_labels = []
for weather_type, flag_col in [('Rainy', 'Is_Rainy'), ('Snowy', 'Is_Snow'), ('Sunny', 'Is_Sunny')]:
    data = df_hourly[df_hourly[flag_col]==1]['Total_Hourly_Revenue'].values
    if len(data) > 0:
        weather_data_for_box.append(data)
        box_labels.append(weather_type)

other_data = df_hourly[other_mask]['Total_Hourly_Revenue'].values
if len(other_data) > 0:
    weather_data_for_box.append(other_data)
    box_labels.append('Other')

box_plot = plt.boxplot(weather_data_for_box, labels=box_labels, patch_artist=True)
for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
plt.title('Revenue Distribution by Weather', fontsize=14, fontweight='bold')
plt.ylabel('Hourly Revenue ($)')
plt.xticks(rotation=45)

# Temperature vs Revenue with Weather Overlay
plt.subplot(4, 3, 10)
for weather_type, flag_col, color in [('Rainy', 'Is_Rainy', 'blue'), 
                                     ('Snowy', 'Is_Snow', 'lightblue'), 
                                     ('Sunny', 'Is_Sunny', 'orange')]:
    mask = df_hourly[flag_col] == 1
    if mask.sum() > 0:
        plt.scatter(df_hourly[mask]['temp_C'], df_hourly[mask]['Total_Hourly_Revenue'], 
                   alpha=0.6, label=weather_type, color=color, s=40)

plt.title('Temperature vs Revenue by Weather Type', fontsize=14, fontweight='bold')
plt.xlabel('Temperature (°C)')
plt.ylabel('Hourly Revenue ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Monthly Revenue Trend
plt.subplot(4, 3, 11)
df_hourly['Month'] = df_hourly['Date'].dt.month
monthly_revenue = df_hourly.groupby('Month')['Total_Hourly_Revenue'].mean()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.plot(monthly_revenue.index, monthly_revenue.values, marker='o', 
         linewidth=3, markersize=10, color='darkgreen')
plt.title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Average Revenue ($)')
plt.xticks(range(1, 13), [month_names[i-1] for i in range(1, 13)], rotation=45)
plt.grid(True, alpha=0.3)

# Weather Impact Summary
plt.subplot(4, 3, 12)
impact_data = []
impact_labels = []
baseline_revenue = df_hourly['Total_Hourly_Revenue'].mean()

for weather_type, flag_col in [('Rainy', 'Is_Rainy'), ('Snowy', 'Is_Snow'), ('Sunny', 'Is_Sunny')]:
    weather_avg = df_hourly[df_hourly[flag_col]==1]['Total_Hourly_Revenue'].mean()
    impact = ((weather_avg - baseline_revenue) / baseline_revenue) * 100
    impact_data.append(impact)
    impact_labels.append(weather_type)

colors_impact = ['red' if x < 0 else 'green' for x in impact_data]
bars = plt.bar(impact_labels, impact_data, color=colors_impact, alpha=0.7)
plt.title('Weather Impact on Revenue (%)', fontsize=14, fontweight='bold')
plt.ylabel('% Change from Average')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -0.5),
             f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
             fontweight='bold')

plt.tight_layout()
plt.show()

# Build regression model
print("\nBuilding regression model...")

# Add lagged variables
df_hourly['Revenue_Lag1'] = df_hourly['Total_Hourly_Revenue'].shift(1)
df_hourly['Revenue_Lag2'] = df_hourly['Total_Hourly_Revenue'].shift(2)
df_hourly = df_hourly.dropna()

print(f"Dataset after adding lags: {df_hourly.shape[0]} rows")

# Define features and target
X = df_hourly[['Revenue_Lag1', 'Revenue_Lag2', 'Is_Rainy', 'Is_Snow', 'Is_Sunny']]
y = df_hourly['Total_Hourly_Revenue']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train model
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model results
print("\nModel Results:")
print("-" * 30)

print("Linear Regression Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:15}: {coef:8.2f}")

print(f"\nIntercept: {model.intercept_:.2f}")

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"\nModel Performance:")
print(f"   R² Score: {r2:.3f} ({r2*100:.1f}% variance explained)")
print(f"   MAE:      ${mae:.2f}")
print(f"   RMSE:     ${rmse:.2f}")

# Model Performance Visualization
plt.figure(figsize=(15, 5))

# Actual vs Predicted
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue', s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Revenue ($)')
plt.ylabel('Predicted Revenue ($)')
plt.title(f'Actual vs Predicted (Boston Weather)\nR² = {r2:.3f}', fontweight='bold')
plt.grid(True, alpha=0.3)

# Residuals Plot
plt.subplot(1, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, color='orange', s=50)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Revenue ($)')
plt.ylabel('Residuals ($)')
plt.title('Residual Analysis', fontweight='bold')
plt.grid(True, alpha=0.3)

# Feature Importance
plt.subplot(1, 3, 3)
feature_importance = abs(model.coef_)
feature_names = X.columns
bars = plt.barh(feature_names, feature_importance, color='lightcoral')
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance (Boston Model)', fontweight='bold')
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print(f"\nBOSTON WEATHER ANALYSIS SUMMARY")
print("=" * 50)
print(f"Location: Boston Weather Data")
print(f"Analysis Period: {df_hourly['Date'].min()} to {df_hourly['Date'].max()}")
print(f"Total Hours: {len(df_hourly)}")
print(f"Avg Temperature: {df_hourly['temp_C'].mean():.1f}°C")
print(f"Rainy Hours: {df_hourly['Is_Rainy'].sum()} ({df_hourly['Is_Rainy'].mean()*100:.1f}%)")
print(f"Snowy Hours: {df_hourly['Is_Snow'].sum()} ({df_hourly['Is_Snow'].mean()*100:.1f}%)")
print(f"Sunny Hours: {df_hourly['Is_Sunny'].sum()} ({df_hourly['Is_Sunny'].mean()*100:.1f}%)")
print(f"Average Revenue: ${df_hourly['Total_Hourly_Revenue'].mean():.2f}/hour")
print(f"Model Performance: {r2*100:.1f}% accuracy, ±${rmse:.2f} error")

weather_impact_summary = {}
for weather_type, flag_col in [('Rainy', 'Is_Rainy'), ('Snowy', 'Is_Snow'), ('Sunny', 'Is_Sunny')]:
    weather_avg = df_hourly[df_hourly[flag_col]==1]['Total_Hourly_Revenue'].mean()
    impact = weather_avg - df_hourly['Total_Hourly_Revenue'].mean()
    weather_impact_summary[weather_type] = impact
    print(f"{weather_type} Weather Impact: {impact:+.2f}$ per hour")

print(f"\nBoston weather analysis complete!")