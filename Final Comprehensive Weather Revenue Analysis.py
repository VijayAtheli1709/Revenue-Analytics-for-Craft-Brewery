#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Comprehensive Weather Revenue Analysis
Multi-Model Approach with Advanced Visualizations
Created on Sun Apr 13 18:47:53 2025
@author: Data Science Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("husl")

print("FINAL COMPREHENSIVE WEATHER REVENUE ANALYSIS")
print("=" * 60)
print("Multi-Model Machine Learning Approach")
print("Advanced Visualizations & Business Insights")
print("=" * 60)

# Load both datasets
print("\nLoading and Processing Data...")

# Original merged data (North Conway weather)
transactions_merged_df = pd.read_excel('Transactions_Weather_Merged (1).xlsx')
print(f"Original transaction data: {transactions_merged_df.shape}")

# Clean original data
transactions_merged_df['Date'] = pd.to_datetime(transactions_merged_df['Date'])
original_df = transactions_merged_df.drop(columns=['Transaction ID', 'Opened', 'Discount Amount', 'Amount', 'Tax', 'Tip',
                                                   'Total Charge', 'Avg_Transaction_Value', 'Tip_Percent'], errors='ignore')
original_df = original_df.drop_duplicates(subset=['Date', 'Hour']).reset_index(drop=True)
original_df.describe()

# Create flags for original (North Conway)
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
print("Processing Boston weather data...")
boston_weather = pd.read_excel('Boston_hourly_weather.xlsx')
boston_weather['datetime'] = pd.to_datetime(boston_weather['date'] + ' ' + boston_weather['time'])
boston_weather['Date'] = pd.to_datetime(boston_weather['datetime'].dt.date)
boston_weather['Hour'] = boston_weather['datetime'].dt.hour
boston_weather = boston_weather.drop(columns=['date', 'time', 'datetime'])
boston_weather.describe()

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

# Comprehensive EDA visualizations
print("\nCreating Advanced Visualizations...")

def create_weather_analysis_plots(df, location_name):
    """Create comprehensive weather analysis plots for a location"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{location_name} Weather Analysis', fontsize=16, fontweight='bold')
    
    # Revenue by Weather Bar Plot
    weather_conditions = ['Rainy', 'Snowy', 'Sunny']
    weather_revenues = []
    for weather_type, flag_col in [('Rainy', 'Is_Rainy'), ('Snowy', 'Is_Snow'), ('Sunny', 'Is_Sunny')]:
        avg_revenue = df[df[flag_col]==1]['Total_Hourly_Revenue'].mean()
        weather_revenues.append(avg_revenue)
    
    bars = axes[0, 0].bar(weather_conditions, weather_revenues, 
                         color=['#3498db', '#e74c3c', '#f39c12'], alpha=0.8)
    axes[0, 0].set_title('Average Revenue by Weather Condition', fontweight='bold')
    axes[0, 0].set_ylabel('Average Hourly Revenue ($)')
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'${height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Weather Distribution
    weather_counts = [df['Is_Rainy'].sum(), df['Is_Snow'].sum(), df['Is_Sunny'].sum()]
    axes[0, 1].pie(weather_counts, labels=weather_conditions, autopct='%1.1f%%', 
                   colors=['#3498db', '#e74c3c', '#f39c12'], startangle=90)
    axes[0, 1].set_title('Weather Distribution', fontweight='bold')
    
    # Temperature vs Revenue
    scatter = axes[0, 2].scatter(df['temp_C'], df['Total_Hourly_Revenue'], 
                                c=df['precip_mm'], cmap='viridis', alpha=0.6, s=50)
    axes[0, 2].set_xlabel('Temperature (°C)')
    axes[0, 2].set_ylabel('Hourly Revenue ($)')
    axes[0, 2].set_title('Temperature vs Revenue', fontweight='bold')
    plt.colorbar(scatter, ax=axes[0, 2], label='Precipitation (mm)')
    
    # Hourly Pattern
    hourly_avg = df.groupby('Hour')['Total_Hourly_Revenue'].mean()
    hourly_std = df.groupby('Hour')['Total_Hourly_Revenue'].std()
    axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', 
                   linewidth=3, markersize=8, color='steelblue')
    axes[1, 0].fill_between(hourly_avg.index, 
                           hourly_avg.values - hourly_std.values,
                           hourly_avg.values + hourly_std.values, 
                           alpha=0.2, color='steelblue')
    axes[1, 0].set_title('Hourly Revenue Pattern', fontweight='bold')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Average Revenue ($)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Weather Impact Box Plot
    weather_data_for_box = []
    box_labels = []
    for weather_type, flag_col in [('Rainy', 'Is_Rainy'), ('Snowy', 'Is_Snow'), ('Sunny', 'Is_Sunny')]:
        data = df[df[flag_col]==1]['Total_Hourly_Revenue'].values
        if len(data) > 0:
            weather_data_for_box.append(data)
            box_labels.append(weather_type)
    
    box_plot = axes[1, 1].boxplot(weather_data_for_box, labels=box_labels, patch_artist=True)
    colors = ['#3498db', '#e74c3c', '#f39c12']
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 1].set_title('Revenue Distribution by Weather', fontweight='bold')
    axes[1, 1].set_ylabel('Hourly Revenue ($)')
    
    # Correlation Heatmap
    corr_data = df[['Total_Hourly_Revenue', 'temp_C', 'precip_mm', 'visibility_km', 'cloudcover']].corr()
    sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.3f', ax=axes[1, 2])
    axes[1, 2].set_title('Weather-Revenue Correlations', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return weather_revenues

# Create analysis for both locations
print("North Conway Analysis:")
nc_weather_revenues = create_weather_analysis_plots(original_df, "North Conway")

print("Boston Analysis:")
boston_weather_revenues = create_weather_analysis_plots(boston_df, "Boston")

# Add datetime for plotting temperature comparison
original_df['Datetime'] = original_df['Date'] + pd.to_timedelta(original_df['Hour'], unit='h')
boston_df['Datetime'] = boston_df['Date'] + pd.to_timedelta(boston_df['Hour'], unit='h')

# Temperature comparison plot
plt.figure(figsize=(15, 6))
plt.plot(original_df['Datetime'], original_df['temp_C'], label='North Conway Weather', alpha=0.7, linewidth=1)
plt.plot(boston_df['Datetime'], boston_df['temp_C'], label='Boston Weather', alpha=0.7, linewidth=1)
plt.title("Temperature Comparison: North Conway vs Boston Weather Data", fontsize=14, fontweight='bold')
plt.xlabel("Date & Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Advanced machine learning models
print("\nRunning Advanced Machine Learning Models...")

def run_comprehensive_models(df, label):
    """Run multiple ML models and return comprehensive results"""
    print(f"\n--- {label} Model Results ---")
    
    X = df[['Revenue_Lag1', 'Revenue_Lag2', 'Is_Rainy', 'Is_Snow', 'Is_Sunny']]
    y = df['Total_Hourly_Revenue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    results = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    results['Linear Regression'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'r2': r2_score(y_test, y_pred_lr),
        'mae': mean_absolute_error(y_test, y_pred_lr),
        'rmse': mean_squared_error(y_test, y_pred_lr, squared=False),
        'cv_score': cross_val_score(lr, X, y, cv=5, scoring='r2').mean()
    }
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=1, max_depth=10)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    results['Random Forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'r2': r2_score(y_test, y_pred_rf),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'rmse': mean_squared_error(y_test, y_pred_rf, squared=False),
        'cv_score': cross_val_score(rf, X, y, cv=5, scoring='r2').mean(),
        'feature_importance': rf.feature_importances_
    }
    
    # Decision Tree
    dt = DecisionTreeRegressor(max_depth=8, random_state=1, min_samples_split=10)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    
    results['Decision Tree'] = {
        'model': dt,
        'predictions': y_pred_dt,
        'r2': r2_score(y_test, y_pred_dt),
        'mae': mean_absolute_error(y_test, y_pred_dt),
        'rmse': mean_squared_error(y_test, y_pred_dt, squared=False),
        'cv_score': cross_val_score(dt, X, y, cv=5, scoring='r2').mean(),
        'feature_importance': dt.feature_importances_
    }
    
    # Print results
    print(f"{'Model':<15} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'CV R²':<8}")
    print("-" * 55)
    for model_name, model_results in results.items():
        print(f"{model_name:<15} {model_results['r2']:<8.3f} {model_results['mae']:<8.2f} "
              f"{model_results['rmse']:<8.2f} {model_results['cv_score']:<8.3f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    print(f"\nBest Model: {best_model_name} (R² = {results[best_model_name]['r2']:.3f})")
    
    return results, X_test, y_test, X.columns

# Run models for both datasets
nc_results, nc_X_test, nc_y_test, feature_names = run_comprehensive_models(original_df, "North Conway")
boston_results, boston_X_test, boston_y_test, _ = run_comprehensive_models(boston_df, "Boston")

# Model comparison visualizations
print("\nCreating Model Comparison Visualizations...")

def create_model_comparison_plots(nc_results, boston_results, nc_X_test, nc_y_test, boston_X_test, boston_y_test, feature_names):
    """Create comprehensive model comparison visualizations"""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('Comprehensive Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # R² Score Comparison
    models = list(nc_results.keys())
    nc_r2_scores = [nc_results[model]['r2'] for model in models]
    boston_r2_scores = [boston_results[model]['r2'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    axes[0, 0].bar(x - width/2, nc_r2_scores, width, label='North Conway', color='skyblue', alpha=0.8)
    axes[0, 0].bar(x + width/2, boston_r2_scores, width, label='Boston', color='lightcoral', alpha=0.8)
    axes[0, 0].set_title('R² Score Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for i, (nc_val, boston_val) in enumerate(zip(nc_r2_scores, boston_r2_scores)):
        axes[0, 0].text(i - width/2, nc_val + 0.01, f'{nc_val:.3f}', ha='center', va='bottom', fontsize=9)
        axes[0, 0].text(i + width/2, boston_val + 0.01, f'{boston_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # MAE Comparison
    nc_mae_scores = [nc_results[model]['mae'] for model in models]
    boston_mae_scores = [boston_results[model]['mae'] for model in models]
    
    axes[0, 1].bar(x - width/2, nc_mae_scores, width, label='North Conway', color='skyblue', alpha=0.8)
    axes[0, 1].bar(x + width/2, boston_mae_scores, width, label='Boston', color='lightcoral', alpha=0.8)
    axes[0, 1].set_title('Mean Absolute Error Comparison', fontweight='bold')
    axes[0, 1].set_ylabel('MAE ($)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cross-Validation Score Comparison
    nc_cv_scores = [nc_results[model]['cv_score'] for model in models]
    boston_cv_scores = [boston_results[model]['cv_score'] for model in models]
    
    axes[0, 2].bar(x - width/2, nc_cv_scores, width, label='North Conway', color='skyblue', alpha=0.8)
    axes[0, 2].bar(x + width/2, boston_cv_scores, width, label='Boston', color='lightcoral', alpha=0.8)
    axes[0, 2].set_title('Cross-Validation R² Comparison', fontweight='bold')
    axes[0, 2].set_ylabel('CV R² Score')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Feature Importance - Random Forest (North Conway)
    if 'feature_importance' in nc_results['Random Forest']:
        importance_nc = nc_results['Random Forest']['feature_importance']
        bars = axes[1, 0].barh(feature_names, importance_nc, color='skyblue', alpha=0.8)
        axes[1, 0].set_title('Feature Importance - Random Forest (North Conway)', fontweight='bold')
        axes[1, 0].set_xlabel('Importance')
        for i, bar in enumerate(bars):
            width_bar = bar.get_width()
            axes[1, 0].text(width_bar + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{width_bar:.3f}', ha='left', va='center', fontsize=9)
    
    # Feature Importance - Random Forest (Boston)
    if 'feature_importance' in boston_results['Random Forest']:
        importance_boston = boston_results['Random Forest']['feature_importance']
        bars = axes[1, 1].barh(feature_names, importance_boston, color='lightcoral', alpha=0.8)
        axes[1, 1].set_title('Feature Importance - Random Forest (Boston)', fontweight='bold')
        axes[1, 1].set_xlabel('Importance')
        for i, bar in enumerate(bars):
            width_bar = bar.get_width()
            axes[1, 1].text(width_bar + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{width_bar:.3f}', ha='left', va='center', fontsize=9)
    
    # Best Model Predictions - North Conway
    best_nc_model = max(nc_results.keys(), key=lambda k: nc_results[k]['r2'])
    best_nc_pred = nc_results[best_nc_model]['predictions']
    axes[1, 2].scatter(nc_y_test, best_nc_pred, alpha=0.6, color='skyblue', s=50)
    axes[1, 2].plot([nc_y_test.min(), nc_y_test.max()], [nc_y_test.min(), nc_y_test.max()], 'r--', lw=2)
    axes[1, 2].set_xlabel('Actual Revenue ($)')
    axes[1, 2].set_ylabel('Predicted Revenue ($)')
    axes[1, 2].set_title(f'Best Model Predictions - North Conway\n{best_nc_model} (R² = {nc_results[best_nc_model]["r2"]:.3f})', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Best Model Predictions - Boston
    best_boston_model = max(boston_results.keys(), key=lambda k: boston_results[k]['r2'])
    best_boston_pred = boston_results[best_boston_model]['predictions']
    axes[2, 0].scatter(boston_y_test, best_boston_pred, alpha=0.6, color='lightcoral', s=50)
    axes[2, 0].plot([boston_y_test.min(), boston_y_test.max()], [boston_y_test.min(), boston_y_test.max()], 'r--', lw=2)
    axes[2, 0].set_xlabel('Actual Revenue ($)')
    axes[2, 0].set_ylabel('Predicted Revenue ($)')
    axes[2, 0].set_title(f'Best Model Predictions - Boston\n{best_boston_model} (R² = {boston_results[best_boston_model]["r2"]:.3f})', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Residuals Analysis - Best Models
    nc_residuals = nc_y_test - best_nc_pred
    boston_residuals = boston_y_test - best_boston_pred
    axes[2, 1].scatter(best_nc_pred, nc_residuals, alpha=0.6, color='skyblue', label='North Conway', s=40)
    axes[2, 1].scatter(best_boston_pred, boston_residuals, alpha=0.6, color='lightcoral', label='Boston', s=40)
    axes[2, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[2, 1].set_xlabel('Predicted Revenue ($)')
    axes[2, 1].set_ylabel('Residuals ($)')
    axes[2, 1].set_title('Residuals Analysis - Best Models', fontweight='bold')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Model Performance Summary Table
    axes[2, 2].axis('off')
    table_data = []
    headers = ['Model', 'Location', 'R²', 'MAE', 'RMSE', 'CV R²']
    
    for model in models:
        table_data.append([model, 'North Conway', f"{nc_results[model]['r2']:.3f}", 
                          f"${nc_results[model]['mae']:.0f}", f"${nc_results[model]['rmse']:.0f}",
                          f"{nc_results[model]['cv_score']:.3f}"])
        table_data.append([model, 'Boston', f"{boston_results[model]['r2']:.3f}", 
                          f"${boston_results[model]['mae']:.0f}", f"${boston_results[model]['rmse']:.0f}",
                          f"{boston_results[model]['cv_score']:.3f}"])
    
    table = axes[2, 2].table(cellText=table_data, colLabels=headers,
                            cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    axes[2, 2].set_title('Model Performance Summary', fontweight='bold', pad=20)
    
    # Color-code best performers
    for i in range(len(table_data)):
        if i % 2 == 0:  # North Conway rows
            table[(i+1, 0)].set_facecolor('#E3F2FD')  # Light blue
        else:  # Boston rows
            table[(i+1, 0)].set_facecolor('#FFEBEE')  # Light red
    
    plt.tight_layout()
    plt.show()

create_model_comparison_plots(nc_results, boston_results, nc_X_test, nc_y_test, boston_X_test, boston_y_test, feature_names)

# Decision tree visualization
print("\nCreating Decision Tree Visualizations...")

def visualize_decision_trees(nc_results, boston_results, feature_names):
    """Create decision tree visualizations for both locations"""
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # North Conway Decision Tree
    nc_dt = nc_results['Decision Tree']['model']
    plot_tree(nc_dt, feature_names=feature_names, filled=True, rounded=True, 
              fontsize=10, ax=axes[0], max_depth=3)
    axes[0].set_title('Decision Tree - North Conway Weather', fontsize=14, fontweight='bold')
    
    # Boston Decision Tree
    boston_dt = boston_results['Decision Tree']['model']
    plot_tree(boston_dt, feature_names=feature_names, filled=True, rounded=True, 
              fontsize=10, ax=axes[1], max_depth=3)
    axes[1].set_title('Decision Tree - Boston Weather', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

visualize_decision_trees(nc_results, boston_results, feature_names)

# Business insights & recommendations
print("\nBUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 50)

# Find best performing models
best_nc_model = max(nc_results.keys(), key=lambda k: nc_results[k]['r2'])
best_boston_model = max(boston_results.keys(), key=lambda k: boston_results[k]['r2'])

print(f"BEST PERFORMING MODELS:")
print(f"   North Conway: {best_nc_model} (R² = {nc_results[best_nc_model]['r2']:.3f})")
print(f"   Boston:       {best_boston_model} (R² = {boston_results[best_boston_model]['r2']:.3f})")

# Overall best model
if nc_results[best_nc_model]['r2'] > boston_results[best_boston_model]['r2']:
    overall_best = f"North Conway {best_nc_model}"
    overall_r2 = nc_results[best_nc_model]['r2']
else:
    overall_best = f"Boston {best_boston_model}"
    overall_r2 = boston_results[best_boston_model]['r2']

print(f"OVERALL CHAMPION: {overall_best} (R² = {overall_r2:.3f})")

# Weather impact analysis
print(f"\nWEATHER IMPACT ANALYSIS:")
for i, (weather_type, impact) in enumerate([('Rainy', 'negative'), ('Snowy', 'positive'), ('Sunny', 'positive')]):
    nc_avg = nc_weather_revenues[i] if i < len(nc_weather_revenues) else 0
    boston_avg = boston_weather_revenues[i] if i < len(boston_weather_revenues) else 0
    print(f"   {weather_type} Weather:")
    print(f"     North Conway: ${nc_avg:.0f}/hour")
    print(f"     Boston:       ${boston_avg:.0f}/hour")
    print(f"     Difference:   ${boston_avg - nc_avg:+.0f}/hour")

# Feature importance insights
print(f"\nFEATURE IMPORTANCE INSIGHTS:")
if 'feature_importance' in nc_results['Random Forest']:
    nc_importance = nc_results['Random Forest']['feature_importance']
    boston_importance = boston_results['Random Forest']['feature_importance']
    
    print("   Top predictive features:")
    for i, feature in enumerate(feature_names):
        print(f"     {feature}:")
        print(f"       North Conway: {nc_importance[i]:.3f}")
        print(f"       Boston:       {boston_importance[i]:.3f}")

# Model recommendations
print(f"\nMODEL RECOMMENDATIONS:")
print(f"   1. Use {overall_best} for production forecasting")
print(f"   2. {overall_r2*100:.1f}% of revenue variance can be explained by weather + historical patterns")
print(f"   3. Weather conditions show significant impact on hourly revenue")
print(f"   4. Historical revenue (lag variables) are strong predictors")

# Business actions
print(f"\nACTIONABLE BUSINESS INSIGHTS:")
print(f"   REVENUE OPTIMIZATION:")
print(f"      • Adjust staffing based on weather forecasts")
print(f"      • Implement weather-based pricing strategies")
print(f"      • Plan inventory based on weather patterns")
print(f"   ")
print(f"   OPERATIONAL PLANNING:")
print(f"      • Use {overall_best} for daily revenue forecasts")
print(f"      • Monitor weather forecasts 24-48 hours ahead")
print(f"      • Adjust marketing campaigns based on weather")
print(f"   ")
print(f"   PERFORMANCE MONITORING:")
print(f"      • Track actual vs predicted revenue daily")
print(f"      • Retrain models quarterly with new data")
print(f"      • Monitor model accuracy degradation")

# Final summary
print(f"\nFINAL ANALYSIS SUMMARY")
print("=" * 50)
print(f"North Conway Data:")
print(f"    • Hours analyzed: {len(original_df)}")
print(f"    • Best model: {best_nc_model}")
print(f"    • Model accuracy: {nc_results[best_nc_model]['r2']*100:.1f}%")
print(f"    • Prediction error: ±${nc_results[best_nc_model]['rmse']:.0f}")

print(f"\nBoston Data:")
print(f"    • Hours analyzed: {len(boston_df)}")
print(f"    • Best model: {best_boston_model}")
print(f"    • Model accuracy: {boston_results[best_boston_model]['r2']*100:.1f}%")
print(f"    • Prediction error: ±${boston_results[best_boston_model]['rmse']:.0f}")

print(f"\nNEXT STEPS:")
print(f"   1. Deploy {overall_best} to production")
print(f"   2. Set up automated weather data feeds")
print(f"   3. Create real-time revenue forecasting dashboard")
print(f"   4. Implement alert system for significant weather events")
print(f"   5. Plan A/B tests for weather-based strategies")

print(f"\nComprehensive analysis complete!")
print(f"Ready for business implementation and deployment!")