#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparative Analysis: North Conway vs Boston Weather Impact on Revenue
Created on Sun Apr 13 18:47:53 2025
@author: Data Analysis Team
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
sns.set_style("whitegrid")
sns.set_palette("Set1")

print("COMPARATIVE WEATHER ANALYSIS: NORTH CONWAY vs BOSTON")
print("=" * 65)

# Load both datasets
print("Loading datasets...")

# North Conway data (original)
transactions_merged_df = pd.read_excel('Transactions_Weather_Merged (1).xlsx')
print(f"Original data (North Conway): {transactions_merged_df.shape}")

# Clean original data
transactions_merged_df['Date'] = pd.to_datetime(transactions_merged_df['Date'])
original_df = transactions_merged_df.drop(columns=['Transaction ID', 'Opened', 'Discount Amount', 'Amount', 'Tax', 'Tip',
                                                   'Total Charge', 'Avg_Transaction_Value', 'Tip_Percent'], errors='ignore')
original_df = original_df.drop_duplicates(subset=['Date', 'Hour']).reset_index(drop=True)

# Create flags for North Conway
original_df['Is_Rainy'] = (original_df['precip_mm'] > 0).astype(int)
original_df['Is_Snow'] = (
    (original_df['precip_mm'] > 0) &
    (original_df['temp_C'] <= 2) &
    (original_df['visibility_km'] <= 5)
).astype(int)
original_df['Is_Sunny'] = (
    (original_df['precip_mm'] == 0) &
    (original_df['cloudcover'] <= 30) &
    (original_df['visibility_km'] >= 8)
).astype(int)
original_df['Revenue_Lag1'] = original_df['Total_Hourly_Revenue'].shift(1)
original_df['Revenue_Lag2'] = original_df['Total_Hourly_Revenue'].shift(2)
original_df = original_df.dropna()

print(f"North Conway processed: {original_df.shape[0]} hours")

# Boston weather version
print("Loading Boston weather data...")
boston_weather = pd.read_excel('Boston_hourly_weather.xlsx')
boston_weather['datetime'] = pd.to_datetime(boston_weather['date'] + ' ' + boston_weather['time'])
boston_weather['Date'] = pd.to_datetime(boston_weather['datetime'].dt.date)
boston_weather['Hour'] = boston_weather['datetime'].dt.hour
boston_weather = boston_weather.drop(columns=['date', 'time', 'datetime'])

# Copy original data structure but merge with Boston weather
boston_df = original_df.drop(columns=['precip_mm', 'temp_C', 'visibility_km', 'cloudcover'], errors='ignore')
boston_df = pd.merge(boston_df, boston_weather, on=['Date', 'Hour'], how='left')

# Create flags for Boston
boston_df['Is_Rainy'] = (boston_df['precip_mm'] > 0).astype(int)
boston_df['Is_Snow'] = (
    (boston_df['precip_mm'] > 0) &
    (boston_df['temp_C'] <= 2) &
    (boston_df['visibility_km'] <= 5)
).astype(int)
boston_df['Is_Sunny'] = (
    (boston_df['precip_mm'] == 0) &
    (boston_df['cloudcover'] <= 30) &
    (boston_df['visibility_km'] >= 8)
).astype(int)
boston_df['Revenue_Lag1'] = boston_df['Total_Hourly_Revenue'].shift(1)
boston_df['Revenue_Lag2'] = boston_df['Total_Hourly_Revenue'].shift(2)
boston_df = boston_df.dropna()

print(f"Boston processed: {boston_df.shape[0]} hours")

# Comparative weather analysis
print("\nWeather comparison analysis...")

def get_weather_stats(df, location_name):
    stats = {
        'Location': location_name,
        'Total_Hours': len(df),
        'Avg_Temp': df['temp_C'].mean(),
        'Avg_Precip': df['precip_mm'].mean(),
        'Avg_Visibility': df['visibility_km'].mean(),
        'Avg_CloudCover': df['cloudcover'].mean(),
        'Rainy_Hours': df['Is_Rainy'].sum(),
        'Snow_Hours': df['Is_Snow'].sum(),
        'Sunny_Hours': df['Is_Sunny'].sum(),
        'Rainy_Pct': df['Is_Rainy'].mean() * 100,
        'Snow_Pct': df['Is_Snow'].mean() * 100,
        'Sunny_Pct': df['Is_Sunny'].mean() * 100,
        'Avg_Revenue': df['Total_Hourly_Revenue'].mean()
    }
    return stats

nc_stats = get_weather_stats(original_df, 'North Conway')
boston_stats = get_weather_stats(boston_df, 'Boston')

# Create comparison dataframe
comparison_df = pd.DataFrame([nc_stats, boston_stats])
print("\nWeather Statistics Comparison:")
print(comparison_df[['Location', 'Avg_Temp', 'Avg_Precip', 'Rainy_Pct', 'Snow_Pct', 'Sunny_Pct', 'Avg_Revenue']].round(2))

# Create comprehensive comparative visualizations
print("\nCreating comparative visualizations...")

fig = plt.figure(figsize=(20, 16))

# Weather Distribution Comparison - Pie Charts
plt.subplot(4, 4, 1)
nc_weather_dist = [nc_stats['Rainy_Hours'], nc_stats['Snow_Hours'], nc_stats['Sunny_Hours'], 
                   nc_stats['Total_Hours'] - nc_stats['Rainy_Hours'] - nc_stats['Snow_Hours'] - nc_stats['Sunny_Hours']]
plt.pie(nc_weather_dist, labels=['Rainy', 'Snow', 'Sunny', 'Other'], autopct='%1.1f%%', startangle=90)
plt.title('North Conway Weather Distribution', fontweight='bold')

plt.subplot(4, 4, 2)
boston_weather_dist = [boston_stats['Rainy_Hours'], boston_stats['Snow_Hours'], boston_stats['Sunny_Hours'],
                       boston_stats['Total_Hours'] - boston_stats['Rainy_Hours'] - boston_stats['Snow_Hours'] - boston_stats['Sunny_Hours']]
plt.pie(boston_weather_dist, labels=['Rainy', 'Snow', 'Sunny', 'Other'], autopct='%1.1f%%', startangle=90)
plt.title('Boston Weather Distribution', fontweight='bold')

# Temperature Comparison
plt.subplot(4, 4, 3)
plt.hist(original_df['temp_C'], bins=30, alpha=0.7, label='North Conway', color='skyblue', density=True)
plt.hist(boston_df['temp_C'], bins=30, alpha=0.7, label='Boston', color='lightcoral', density=True)
plt.title('Temperature Distribution Comparison', fontweight='bold')
plt.xlabel('Temperature (°C)')
plt.ylabel('Density')
plt.legend()

# Precipitation Comparison
plt.subplot(4, 4, 4)
plt.hist(original_df['precip_mm'], bins=30, alpha=0.7, label='North Conway', color='blue', density=True)
plt.hist(boston_df['precip_mm'], bins=30, alpha=0.7, label='Boston', color='red', density=True)
plt.title('Precipitation Distribution', fontweight='bold')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Density')
plt.legend()

# Revenue by Weather - Side by Side
plt.subplot(4, 4, 5)
weather_types = ['Rainy', 'Snow', 'Sunny', 'Other']
nc_revenues = []
boston_revenues = []

for weather_type, flag_col in [('Rainy', 'Is_Rainy'), ('Snow', 'Is_Snow'), ('Sunny', 'Is_Sunny')]:
    nc_rev = original_df[original_df[flag_col]==1]['Total_Hourly_Revenue'].mean()
    boston_rev = boston_df[boston_df[flag_col]==1]['Total_Hourly_Revenue'].mean()
    nc_revenues.append(nc_rev)
    boston_revenues.append(boston_rev)

# Other weather
nc_other = original_df[(original_df['Is_Rainy']==0) & (original_df['Is_Snow']==0) & (original_df['Is_Sunny']==0)]['Total_Hourly_Revenue'].mean()
boston_other = boston_df[(boston_df['Is_Rainy']==0) & (boston_df['Is_Snow']==0) & (boston_df['Is_Sunny']==0)]['Total_Hourly_Revenue'].mean()
nc_revenues.append(nc_other)
boston_revenues.append(boston_other)

x = np.arange(len(weather_types))
width = 0.35
plt.bar(x - width/2, nc_revenues, width, label='North Conway', color='skyblue')
plt.bar(x + width/2, boston_revenues, width, label='Boston', color='lightcoral')
plt.title('Revenue by Weather Type', fontweight='bold')
plt.xlabel('Weather Type')
plt.ylabel('Average Revenue ($)')
plt.xticks(x, weather_types, rotation=45)
plt.legend()

# Temperature vs Revenue Comparison
plt.subplot(4, 4, 6)
plt.scatter(original_df['temp_C'], original_df['Total_Hourly_Revenue'], 
           alpha=0.5, label='North Conway', color='blue', s=20)
plt.scatter(boston_df['temp_C'], boston_df['Total_Hourly_Revenue'], 
           alpha=0.5, label='Boston', color='red', s=20)
plt.title('Temperature vs Revenue', fontweight='bold')
plt.xlabel('Temperature (°C)')
plt.ylabel('Revenue ($)')
plt.legend()

# Hourly Pattern Comparison
plt.subplot(4, 4, 7)
nc_hourly = original_df.groupby('Hour')['Total_Hourly_Revenue'].mean()
boston_hourly = boston_df.groupby('Hour')['Total_Hourly_Revenue'].mean()
plt.plot(nc_hourly.index, nc_hourly.values, marker='o', label='North Conway', linewidth=2)
plt.plot(boston_hourly.index, boston_hourly.values, marker='s', label='Boston', linewidth=2)
plt.title('Hourly Revenue Pattern', fontweight='bold')
plt.xlabel('Hour of Day')
plt.ylabel('Average Revenue ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Weather Impact Comparison
plt.subplot(4, 4, 8)
nc_baseline = original_df['Total_Hourly_Revenue'].mean()
boston_baseline = boston_df['Total_Hourly_Revenue'].mean()

nc_impacts = []
boston_impacts = []
for weather_type, flag_col in [('Rainy', 'Is_Rainy'), ('Snow', 'Is_Snow'), ('Sunny', 'Is_Sunny')]:
    nc_impact = ((original_df[original_df[flag_col]==1]['Total_Hourly_Revenue'].mean() - nc_baseline) / nc_baseline) * 100
    boston_impact = ((boston_df[boston_df[flag_col]==1]['Total_Hourly_Revenue'].mean() - boston_baseline) / boston_baseline) * 100
    nc_impacts.append(nc_impact)
    boston_impacts.append(boston_impact)

weather_labels = ['Rainy', 'Snow', 'Sunny']
x = np.arange(len(weather_labels))
plt.bar(x - width/2, nc_impacts, width, label='North Conway', color='skyblue')
plt.bar(x + width/2, boston_impacts, width, label='Boston', color='lightcoral')
plt.title('Weather Impact on Revenue (%)', fontweight='bold')
plt.xlabel('Weather Type')
plt.ylabel('% Change from Average')
plt.xticks(x, weather_labels)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.legend()

# Correlation Heatmap Comparison
plt.subplot(4, 4, 9)
nc_corr = original_df[['Total_Hourly_Revenue', 'temp_C', 'precip_mm', 'visibility_km', 'cloudcover']].corr()
sns.heatmap(nc_corr, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f', cbar=False)
plt.title('North Conway Correlations', fontweight='bold')

plt.subplot(4, 4, 10)
boston_corr = boston_df[['Total_Hourly_Revenue', 'temp_C', 'precip_mm', 'visibility_km', 'cloudcover']].corr()
sns.heatmap(boston_corr, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f', cbar=False)
plt.title('Boston Correlations', fontweight='bold')

# Revenue Distribution Comparison
plt.subplot(4, 4, 11)
plt.hist(original_df['Total_Hourly_Revenue'], bins=30, alpha=0.7, label='North Conway', density=True, color='skyblue')
plt.hist(boston_df['Total_Hourly_Revenue'], bins=30, alpha=0.7, label='Boston', density=True, color='lightcoral')
plt.title('Revenue Distribution Comparison', fontweight='bold')
plt.xlabel('Hourly Revenue ($)')
plt.ylabel('Density')
plt.legend()

# Visibility Comparison
plt.subplot(4, 4, 12)
plt.hist(original_df['visibility_km'], bins=30, alpha=0.7, label='North Conway', density=True, color='green')
plt.hist(boston_df['visibility_km'], bins=30, alpha=0.7, label='Boston', density=True, color='orange')
plt.title('Visibility Distribution', fontweight='bold')
plt.xlabel('Visibility (km)')
plt.ylabel('Density')
plt.legend()

# Monthly Comparison
plt.subplot(4, 4, 13)
original_df['Month'] = original_df['Date'].dt.month
boston_df['Month'] = boston_df['Date'].dt.month
nc_monthly = original_df.groupby('Month')['Total_Hourly_Revenue'].mean()
boston_monthly = boston_df.groupby('Month')['Total_Hourly_Revenue'].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.plot(nc_monthly.index, nc_monthly.values, marker='o', label='North Conway', linewidth=2)
plt.plot(boston_monthly.index, boston_monthly.values, marker='s', label='Boston', linewidth=2)
plt.title('Monthly Revenue Trends', fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Average Revenue ($)')
plt.xticks(range(1, 13), [months[i-1] for i in range(1, 13)], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Cloud Cover Comparison
plt.subplot(4, 4, 14)
plt.scatter(original_df['cloudcover'], original_df['Total_Hourly_Revenue'], 
           alpha=0.5, label='North Conway', color='purple', s=20)
plt.scatter(boston_df['cloudcover'], boston_df['Total_Hourly_Revenue'], 
           alpha=0.5, label='Boston', color='brown', s=20)
plt.title('Cloud Cover vs Revenue', fontweight='bold')
plt.xlabel('Cloud Cover (%)')
plt.ylabel('Revenue ($)')
plt.legend()

# Weather Statistics Comparison Table
plt.subplot(4, 4, 15)
plt.axis('off')
table_data = []
metrics = ['Avg Temp (°C)', 'Avg Precip (mm)', 'Rainy %', 'Snow %', 'Sunny %', 'Avg Revenue ($)']
nc_values = [f"{nc_stats['Avg_Temp']:.1f}", f"{nc_stats['Avg_Precip']:.2f}", 
             f"{nc_stats['Rainy_Pct']:.1f}", f"{nc_stats['Snow_Pct']:.1f}", 
             f"{nc_stats['Sunny_Pct']:.1f}", f"{nc_stats['Avg_Revenue']:.0f}"]
boston_values = [f"{boston_stats['Avg_Temp']:.1f}", f"{boston_stats['Avg_Precip']:.2f}", 
                 f"{boston_stats['Rainy_Pct']:.1f}", f"{boston_stats['Snow_Pct']:.1f}", 
                 f"{boston_stats['Sunny_Pct']:.1f}", f"{boston_stats['Avg_Revenue']:.0f}"]

for i, metric in enumerate(metrics):
    table_data.append([metric, nc_values[i], boston_values[i]])

table = plt.table(cellText=table_data, colLabels=['Metric', 'North Conway', 'Boston'],
                  cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title('Weather Statistics Comparison', fontweight='bold', pad=20)

# Revenue Volatility Comparison
plt.subplot(4, 4, 16)
nc_volatility = original_df.groupby('Hour')['Total_Hourly_Revenue'].std()
boston_volatility = boston_df.groupby('Hour')['Total_Hourly_Revenue'].std()
plt.plot(nc_volatility.index, nc_volatility.values, marker='o', label='North Conway', linewidth=2)
plt.plot(boston_volatility.index, boston_volatility.values, marker='s', label='Boston', linewidth=2)
plt.title('Revenue Volatility by Hour', fontweight='bold')
plt.xlabel('Hour of Day')
plt.ylabel('Revenue Std Dev ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Run regression for both datasets
print("\nRunning comparative regression analysis...")

def run_regression(df, label):
    X = df[['Revenue_Lag1', 'Revenue_Lag2', 'Is_Rainy', 'Is_Snow', 'Is_Sunny']]
    y = df['Total_Hourly_Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results = {
        'model': model,
        'y_test': y_test,
        'y_pred': y_pred,
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': mean_squared_error(y_test, y_pred, squared=False),
        'coefficients': dict(zip(X.columns, model.coef_)),
        'intercept': model.intercept_
    }
    
    print(f"\n--- {label} Regression Results ---")
    print("Coefficients:")
    for feature, coef in results['coefficients'].items():
        print(f"  {feature:15}: {coef:8.2f}")
    print(f"Intercept: {results['intercept']:.2f}")
    print(f"R²: {results['r2']:.3f} ({results['r2']*100:.1f}% variance explained)")
    print(f"MAE: ${results['mae']:.2f}")
    print(f"RMSE: ${results['rmse']:.2f}")
    
    return results

nc_results = run_regression(original_df, "North Conway Weather")
boston_results = run_regression(boston_df, "Boston Weather")

# Model comparison visualization
print("\nCreating model comparison visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Model Performance Comparison
model_comparison = pd.DataFrame({
    'North Conway': [nc_results['r2'], nc_results['mae'], nc_results['rmse']],
    'Boston': [boston_results['r2'], boston_results['mae'], boston_results['rmse']]
}, index=['R²', 'MAE ($)', 'RMSE ($)'])

# R² Score Comparison
axes[0, 0].bar(model_comparison.columns, model_comparison.loc['R²'], color=['skyblue', 'lightcoral'])
axes[0, 0].set_title('R² Score Comparison', fontweight='bold')
axes[0, 0].set_ylabel('R² Score')
for i, v in enumerate(model_comparison.loc['R²']):
    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Error Comparison
x = np.arange(2)
width = 0.35
axes[0, 1].bar(x - width/2, [nc_results['mae'], nc_results['rmse']], width, label='North Conway', color='skyblue')
axes[0, 1].bar(x + width/2, [boston_results['mae'], boston_results['rmse']], width, label='Boston', color='lightcoral')
axes[0, 1].set_title('Error Metrics Comparison', fontweight='bold')
axes[0, 1].set_ylabel('Error ($)')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(['MAE', 'RMSE'])
axes[0, 1].legend()

# Coefficient Comparison
features = list(nc_results['coefficients'].keys())
nc_coefs = [nc_results['coefficients'][f] for f in features]
boston_coefs = [boston_results['coefficients'][f] for f in features]

x = np.arange(len(features))
axes[0, 2].bar(x - width/2, nc_coefs, width, label='North Conway', color='skyblue')
axes[0, 2].bar(x + width/2, boston_coefs, width, label='Boston', color='lightcoral')
axes[0, 2].set_title('Coefficient Comparison', fontweight='bold')
axes[0, 2].set_ylabel('Coefficient Value')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(features, rotation=45, ha='right')
axes[0, 2].legend()
axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Actual vs Predicted - North Conway
axes[1, 0].scatter(nc_results['y_test'], nc_results['y_pred'], alpha=0.6, color='skyblue')
axes[1, 0].plot([nc_results['y_test'].min(), nc_results['y_test'].max()], 
                [nc_results['y_test'].min(), nc_results['y_test'].max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual Revenue ($)')
axes[1, 0].set_ylabel('Predicted Revenue ($)')
axes[1, 0].set_title(f'North Conway: Actual vs Predicted\nR² = {nc_results["r2"]:.3f}', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Actual vs Predicted - Boston
axes[1, 1].scatter(boston_results['y_test'], boston_results['y_pred'], alpha=0.6, color='lightcoral')
axes[1, 1].plot([boston_results['y_test'].min(), boston_results['y_test'].max()], 
                [boston_results['y_test'].min(), boston_results['y_test'].max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual Revenue ($)')
axes[1, 1].set_ylabel('Predicted Revenue ($)')
axes[1, 1].set_title(f'Boston: Actual vs Predicted\nR² = {boston_results["r2"]:.3f}', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Residuals Comparison
nc_residuals = nc_results['y_test'] - nc_results['y_pred']
boston_residuals = boston_results['y_test'] - boston_results['y_pred']
axes[1, 2].scatter(nc_results['y_pred'], nc_residuals, alpha=0.6, color='skyblue', label='North Conway')
axes[1, 2].scatter(boston_results['y_pred'], boston_residuals, alpha=0.6, color='lightcoral', label='Boston')
axes[1, 2].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 2].set_xlabel('Predicted Revenue ($)')
axes[1, 2].set_ylabel('Residuals ($)')
axes[1, 2].set_title('Residuals Comparison', fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Detailed comparison summary
print(f"\nCOMPREHENSIVE COMPARISON SUMMARY")
print("=" * 60)

print(f"\nNORTH CONWAY WEATHER DATA:")
print(f"   Period: {original_df['Date'].min()} to {original_df['Date'].max()}")
print(f"   Hours: {len(original_df)}")
print(f"   Avg Temp: {nc_stats['Avg_Temp']:.1f}°C")
print(f"   Precipitation: {nc_stats['Avg_Precip']:.2f}mm avg")
print(f"   Rainy: {nc_stats['Rainy_Pct']:.1f}% | Snow: {nc_stats['Snow_Pct']:.1f}% | Sunny: {nc_stats['Sunny_Pct']:.1f}%")
print(f"   Avg Revenue: ${nc_stats['Avg_Revenue']:.2f}/hour")
print(f"   Model Performance: R²={nc_results['r2']:.3f}, MAE=${nc_results['mae']:.2f}, RMSE=${nc_results['rmse']:.2f}")

print(f"\nBOSTON WEATHER DATA:")
print(f"   Period: {boston_df['Date'].min()} to {boston_df['Date'].max()}")
print(f"   Hours: {len(boston_df)}")
print(f"   Avg Temp: {boston_stats['Avg_Temp']:.1f}°C")
print(f"   Precipitation: {boston_stats['Avg_Precip']:.2f}mm avg")
print(f"   Rainy: {boston_stats['Rainy_Pct']:.1f}% | Snow: {boston_stats['Snow_Pct']:.1f}% | Sunny: {boston_stats['Sunny_Pct']:.1f}%")
print(f"   Avg Revenue: ${boston_stats['Avg_Revenue']:.2f}/hour")
print(f"   Model Performance: R²={boston_results['r2']:.3f}, MAE=${boston_results['mae']:.2f}, RMSE=${boston_results['rmse']:.2f}")

print(f"\nKEY DIFFERENCES:")
temp_diff = boston_stats['Avg_Temp'] - nc_stats['Avg_Temp']
precip_diff = boston_stats['Avg_Precip'] - nc_stats['Avg_Precip']
revenue_diff = boston_stats['Avg_Revenue'] - nc_stats['Avg_Revenue']
r2_diff = boston_results['r2'] - nc_results['r2']

print(f"   Temperature: Boston is {temp_diff:+.1f}°C {'warmer' if temp_diff > 0 else 'cooler'}")
print(f"   Precipitation: Boston has {precip_diff:+.2f}mm {'more' if precip_diff > 0 else 'less'} rain")
print(f"   Revenue: Boston data shows ${revenue_diff:+.2f}/hour {'higher' if revenue_diff > 0 else 'lower'} average")
print(f"   Model Accuracy: Boston model is {r2_diff:+.3f} {'better' if r2_diff > 0 else 'worse'} (R² difference)")

print(f"\nWEATHER IMPACT WINNERS:")
for weather_type, flag_col in [('Rainy', 'Is_Rainy'), ('Snowy', 'Is_Snow'), ('Sunny', 'Is_Sunny')]:
    nc_impact = ((original_df[original_df[flag_col]==1]['Total_Hourly_Revenue'].mean() - nc_stats['Avg_Revenue']) / nc_stats['Avg_Revenue']) * 100
    boston_impact = ((boston_df[boston_df[flag_col]==1]['Total_Hourly_Revenue'].mean() - boston_stats['Avg_Revenue']) / boston_stats['Avg_Revenue']) * 100
    
    if abs(nc_impact) > abs(boston_impact):
        winner = "North Conway"
        impact_value = nc_impact
    else:
        winner = "Boston"
        impact_value = boston_impact
    
    print(f"   {weather_type}: {winner} shows stronger impact ({impact_value:+.1f}%)")

print(f"\nMODEL PERFORMANCE COMPARISON:")
if nc_results['r2'] > boston_results['r2']:
    better_model = "North Conway"
    r2_advantage = nc_results['r2'] - boston_results['r2']
else:
    better_model = "Boston"
    r2_advantage = boston_results['r2'] - nc_results['r2']

print(f"   Better R²: {better_model} (+{r2_advantage:.3f})")

if nc_results['mae'] < boston_results['mae']:
    lower_mae = "North Conway"
    mae_advantage = boston_results['mae'] - nc_results['mae']
else:
    lower_mae = "Boston"
    mae_advantage = nc_results['mae'] - boston_results['mae']

print(f"   Lower MAE: {lower_mae} (-${mae_advantage:.2f})")

print(f"\nINSIGHTS & RECOMMENDATIONS:")
print(f"   1. {'Boston' if boston_results['r2'] > nc_results['r2'] else 'North Conway'} weather data provides better predictive accuracy")
print(f"   2. Weather patterns differ significantly between locations")
print(f"   3. Temperature difference: {abs(temp_diff):.1f}°C may affect customer behavior")
print(f"   4. Location-specific weather data improves model performance")

# Feature importance comparison
print(f"\nFEATURE IMPORTANCE COMPARISON:")
print(f"{'Feature':<15} {'North Conway':<12} {'Boston':<12} {'Difference':<12}")
print("-" * 55)
for feature in features:
    nc_coef = nc_results['coefficients'][feature]
    boston_coef = boston_results['coefficients'][feature]
    diff = boston_coef - nc_coef
    print(f"{feature:<15} {nc_coef:>8.3f}     {boston_coef:>8.3f}     {diff:>+8.3f}")

print(f"\nComparative analysis complete!")
print(f"Consider using {'Boston' if boston_results['r2'] > nc_results['r2'] else 'North Conway'} weather data for production models.")