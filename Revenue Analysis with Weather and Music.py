import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from scipy import stats

# Import Required Libraries
transactions = pd.read_excel("Transactions_Weather_Merged (1).xlsx")
music = pd.read_excel("live-music-data-v1.xlsx")

# Inspect Column Names
print(transactions.columns)
print(music.columns)

# Ensure Consistent Date Formats
transactions['Date'] = pd.to_datetime(transactions['Date'])
music['Date'] = pd.to_datetime(music['Date'])

# Calculate Daily Revenue
daily_revenue = transactions.groupby('Date')['Total Charge'].sum().reset_index()
daily_revenue.rename(columns={'Total Charge': 'total_revenue'}, inplace=True)

# Flag Live Music Days
music_days = music[['Date']].drop_duplicates()
music_days['has_music'] = 1

# Merge Revenue with Music Flags
daily_revenue = pd.merge(daily_revenue, music_days, on='Date', how='left')
daily_revenue['has_music'] = daily_revenue['has_music'].fillna(0).astype(int)

# Compare Average Revenue
revenue_summary = daily_revenue.groupby('has_music')['total_revenue'].agg(['mean', 'count']).reset_index()
print("Basic Revenue Summary by Music Presence:")
print(revenue_summary)

# Visualize Revenue Differences
plt.figure(figsize=(10, 6))
sns.boxplot(x='has_music', y='total_revenue', data=daily_revenue, palette=['orange', 'steelblue'])
plt.xticks([0, 1], ['No Music', 'Live Music'])
plt.title('Revenue: Days With vs Without Live Music')
plt.xlabel('Live Music')
plt.ylabel('Total Daily Revenue')
plt.savefig('basic_music_revenue_comparison.png')
plt.close()

# Apply weather conditions
# Assuming df_hourly is from your Transactions_Weather_Merged.xlsx
df_hourly = transactions  # Rename this if your hourly weather data is different

# Create snow and sunny indicators
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

# Aggregate weather conditions to daily level
daily_weather = df_hourly.groupby('Date')[['Is_Snow', 'Is_Sunny']].max().reset_index()

# Merge weather conditions with daily revenue
daily_revenue = pd.merge(daily_revenue, daily_weather, on='Date', how='left')

# Fill any missing weather values
daily_revenue['Is_Snow'] = daily_revenue['Is_Snow'].fillna(0).astype(int)
daily_revenue['Is_Sunny'] = daily_revenue['Is_Sunny'].fillna(0).astype(int)

# Create combination categories
daily_revenue['Weather_Music_Category'] = 'Other'
daily_revenue.loc[(daily_revenue['Is_Sunny'] == 1) & (daily_revenue['has_music'] == 1), 'Weather_Music_Category'] = 'Sunny with Music'
daily_revenue.loc[(daily_revenue['Is_Sunny'] == 1) & (daily_revenue['has_music'] == 0), 'Weather_Music_Category'] = 'Sunny without Music'
daily_revenue.loc[(daily_revenue['Is_Snow'] == 1) & (daily_revenue['has_music'] == 1), 'Weather_Music_Category'] = 'Snow with Music'
daily_revenue.loc[(daily_revenue['Is_Snow'] == 1) & (daily_revenue['has_music'] == 0), 'Weather_Music_Category'] = 'Snow without Music'

# Calculate revenue statistics for each category
weather_music_summary = daily_revenue.groupby('Weather_Music_Category')['total_revenue'].agg(['mean', 'median', 'count']).reset_index()
print("\nRevenue Summary by Weather and Music Categories:")
print(weather_music_summary)

# Create visualization comparing these categories
plt.figure(figsize=(12, 6))
sns.boxplot(x='Weather_Music_Category', y='total_revenue', data=daily_revenue)
plt.title('Revenue by Weather and Music Presence')
plt.xlabel('Category')
plt.ylabel('Total Daily Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('weather_music_revenue_comparison.png')
plt.close()

# Statistical comparison between groups
# Perform ANOVA to test if differences between groups are significant
relevant_categories = ['Sunny with Music', 'Sunny without Music', 'Snow with Music', 'Snow without Music']
filtered_data = daily_revenue[daily_revenue['Weather_Music_Category'].isin(relevant_categories)]

# Create lists of revenue values for each group
groups = [filtered_data[filtered_data['Weather_Music_Category'] == cat]['total_revenue'] for cat in relevant_categories]

# Run ANOVA if all groups have data
non_empty_groups = [g for g in groups if len(g) > 0]
if len(non_empty_groups) > 1:
    f_stat, p_value = stats.f_oneway(*non_empty_groups)
    print(f"\nANOVA results: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")

# Calculate percentage impact
try:
    sunny_with = weather_music_summary[weather_music_summary['Weather_Music_Category'] == 'Sunny with Music']['mean'].values[0]
    sunny_without = weather_music_summary[weather_music_summary['Weather_Music_Category'] == 'Sunny without Music']['mean'].values[0]
    sunny_impact = ((sunny_with - sunny_without) / sunny_without) * 100
    print(f"\nMusic impact on sunny days: {sunny_impact:.1f}% change in revenue")
except (IndexError, ZeroDivisionError):
    print("\nNot enough data to calculate impact on sunny days")

try:
    snow_with = weather_music_summary[weather_music_summary['Weather_Music_Category'] == 'Snow with Music']['mean'].values[0]
    snow_without = weather_music_summary[weather_music_summary['Weather_Music_Category'] == 'Snow without Music']['mean'].values[0]
    snow_impact = ((snow_with - snow_without) / snow_without) * 100
    print(f"Music impact on snowy days: {snow_impact:.1f}% change in revenue")
except (IndexError, ZeroDivisionError):
    print("Not enough data to calculate impact on snowy days")

# Time-based analysis
daily_revenue['Day_Of_Week'] = daily_revenue['Date'].dt.day_name()
daily_revenue['Month'] = daily_revenue['Date'].dt.month_name()

# Day of week analysis
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_data = daily_revenue.pivot_table(
    values='total_revenue', 
    index='Day_Of_Week', 
    columns='has_music',
    aggfunc='mean'
).reindex(day_order)

plt.figure(figsize=(10, 6))
try:
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlGnBu')
    plt.title('Average Revenue by Day of Week and Music Presence')
    plt.xlabel('Has Live Music (0=No, 1=Yes)')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.savefig('day_of_week_music_heatmap.png')
except ValueError as e:
    print(f"Could not create heatmap: {e}")
plt.close()

# Monthly analysis
plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='total_revenue', hue='has_music', data=daily_revenue, 
           order=['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December'])
plt.title('Monthly Revenue by Music Presence')
plt.xlabel('Month')
plt.ylabel('Daily Revenue')
plt.xticks(rotation=45)
plt.legend(title='Live Music', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('monthly_music_revenue.png')
plt.close()

# Regression analysis to control for multiple factors
# Create dummy variables for days of week
day_dummies = pd.get_dummies(daily_revenue['Day_Of_Week'], prefix='day')
daily_revenue = pd.concat([daily_revenue, day_dummies], axis=1)

# Run regression
X = daily_revenue[['has_music', 'Is_Sunny', 'Is_Snow'] + list(day_dummies.columns[:-1])]  # Drop one category to avoid multicollinearity
y = daily_revenue['total_revenue']
X = sm.add_constant(X)  # Add intercept

print(X.dtypes)
print(X.head())
X = X.astype(float)
y = pd.to_numeric(y)
model = sm.OLS(y, X).fit()
print("\nRegression Analysis Results:")
print(model.summary())

# Interactive effects analysis
daily_revenue['Music_X_Sunny'] = daily_revenue['has_music'] * daily_revenue['Is_Sunny']
daily_revenue['Music_X_Snow'] = daily_revenue['has_music'] * daily_revenue['Is_Snow']

X_interaction = sm.add_constant(pd.concat([
    daily_revenue[['has_music', 'Is_Sunny', 'Is_Snow']],
    daily_revenue[['Music_X_Sunny', 'Music_X_Snow']],
    day_dummies.iloc[:, :-1]  # Drop one day for multicollinearity
], axis=1))

print(X_interaction.dtypes)
X_interaction = X_interaction.astype(float)
interaction_model = sm.OLS(y, X_interaction).fit()
print("\nInteraction Effects Analysis:")
print(interaction_model.summary())

# Export the enhanced dataset for further analysis
daily_revenue.to_csv('enhanced_revenue_analysis.csv', index=False)

print("\nAnalysis complete. All visualizations have been saved and the enhanced dataset has been exported to 'enhanced_revenue_analysis.csv'")
