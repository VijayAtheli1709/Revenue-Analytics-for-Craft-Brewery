import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from scipy import stats

# Load and prepare the data
transactions = pd.read_excel("Transactions_Weather_Merged.xlsx")
music_raw = pd.read_excel("live-music-data-v1.xlsx")

# Clean and format dates in music data
music_raw['Date'] = pd.to_datetime(music_raw['Date'], format='%d-%m-%Y', errors='coerce')

# Clean column names
music = music_raw.rename(columns={
    'Musical Act': 'Band_ID',
    'Musical Arts.Rate': 'Rate',
    'Musical Arts.Genre/Description': 'Genre'
})

# Ensure consistent date formats
transactions['Date'] = pd.to_datetime(transactions['Date'])

# Calculate daily revenue
daily_revenue = transactions.groupby('Date')['Total Charge'].sum().reset_index()
daily_revenue.rename(columns={'Total Charge': 'total_revenue'}, inplace=True)

# Flag live music days
music_days = music[['Date']].drop_duplicates()
music_days['has_music'] = 1

# Merge revenue with music flags
daily_revenue = pd.merge(daily_revenue, music_days, on='Date', how='left')
daily_revenue['has_music'] = daily_revenue['has_music'].fillna(0).astype(int)

# Add weekday information
daily_revenue['weekday'] = daily_revenue['Date'].dt.day_name()
daily_revenue['is_weekend'] = daily_revenue['weekday'].isin(['Friday', 'Saturday', 'Sunday']).astype(int)

# Create weekend and music combined variable
daily_revenue['weekend_music'] = daily_revenue.apply(
    lambda x: 'Weekend with Music' if x['is_weekend'] == 1 and x['has_music'] == 1 
    else 'Weekend no Music' if x['is_weekend'] == 1 and x['has_music'] == 0
    else 'Weekday with Music' if x['is_weekend'] == 0 and x['has_music'] == 1
    else 'Weekday no Music', axis=1)

# Basic revenue statistics by music presence
revenue_summary = daily_revenue.groupby('has_music')['total_revenue'].agg(['mean', 'median', 'std', 'count']).reset_index()
print("Revenue Summary by Music Presence:")
print(revenue_summary)

# Figure 1: Revenue by Music and Weekend Status
plt.figure(figsize=(12, 8))
sns.boxplot(x='weekend_music', y='total_revenue', data=daily_revenue, 
            order=['Weekday no Music', 'Weekday with Music', 'Weekend no Music', 'Weekend with Music'],
            palette='viridis')
plt.title('Revenue by Live Music and Weekend Status', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Total Daily Revenue', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('revenue_by_music_weekend.png')
plt.show()

# Calculate average revenue for each category
category_avg = daily_revenue.groupby('weekend_music')['total_revenue'].agg(['mean', 'median', 'count']).reset_index()
print("\nAverage Revenue by Category:")
print(category_avg)

# Add band and genre information to daily revenue
music_with_details = music[['Date', 'Band_ID', 'Genre', 'Rate']].drop_duplicates()
daily_revenue_with_bands = pd.merge(daily_revenue, music_with_details, on='Date', how='left')

# Figure 2: Revenue by Genre Category
# First, extract primary genre from the description
def extract_primary_genre(genre_desc):
    if pd.isna(genre_desc):
        return 'Unknown'
    
    genre_lower = genre_desc.lower()
    if 'rock' in genre_lower:
        return 'Rock'
    elif 'blues' in genre_lower:
        return 'Blues'
    elif 'jazz' in genre_lower:
        return 'Jazz'
    elif 'funk' in genre_lower:
        return 'Funk'
    elif 'folk' in genre_lower:
        return 'Folk'
    elif 'country' in genre_lower:
        return 'Country'
    elif 'dj' in genre_lower:
        return 'DJ'
    elif 'solo' in genre_lower:
        return 'Solo'
    else:
        return 'Other'

daily_revenue_with_bands['primary_genre'] = daily_revenue_with_bands['Genre'].apply(extract_primary_genre)

# Analyze revenue by genre
genre_revenue = daily_revenue_with_bands.dropna(subset=['primary_genre'])
genre_revenue = genre_revenue[genre_revenue['primary_genre'] != 'Unknown']

plt.figure(figsize=(14, 8))
sns.boxplot(x='primary_genre', y='total_revenue', data=genre_revenue, palette='viridis')
plt.title('Revenue by Music Genre', fontsize=16)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Total Daily Revenue', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('revenue_by_genre.png')
plt.show()

# Figure 3: Top Bands by Average Revenue
bands_data = daily_revenue_with_bands.dropna(subset=['Band_ID'])
band_revenue = bands_data.groupby('Band_ID').agg(
    avg_revenue=('total_revenue', 'mean'),
    performances=('Band_ID', 'count'),
    genre=('Genre', 'first')
).reset_index()

# Filter to bands with at least 3 performances for more reliable data
frequent_bands = band_revenue[band_revenue['performances'] >= 3].sort_values('avg_revenue', ascending=False)

plt.figure(figsize=(14, 10))
top_bands = frequent_bands.head(10)
bars = sns.barplot(x='avg_revenue', y='Band_ID', data=top_bands, palette='viridis')

# Add performance count and genre annotations
for i, (value, count, genre) in enumerate(zip(top_bands['avg_revenue'], top_bands['performances'], top_bands['genre'])):
    genre_text = str(genre) if not pd.isna(genre) else 'Unknown'
    bars.text(value + 100, i, f"({count} shows, {genre_text})", va='center')

plt.title('Top 10 Bands by Average Daily Revenue', fontsize=16)
plt.xlabel('Average Daily Revenue', fontsize=14)
plt.ylabel('Band ID', fontsize=14)
plt.tight_layout()
plt.savefig('top_bands_revenue.png')
plt.show()

# Figure 4: Revenue by Day of Week with Music Status
plt.figure(figsize=(14, 8))
sns.boxplot(x='weekday', y='total_revenue', hue='has_music', 
            data=daily_revenue, 
            order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            palette=['coral', 'mediumseagreen'])
plt.title('Revenue by Day of Week With/Without Music', fontsize=16)
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Total Daily Revenue', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Has Live Music', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('revenue_by_day_music.png')
plt.show()

# Figure 5: Revenue trend over time with music events marked
daily_revenue_sorted = daily_revenue.sort_values('Date')
plt.figure(figsize=(16, 8))
plt.plot(daily_revenue_sorted['Date'], daily_revenue_sorted['total_revenue'], 'b-', alpha=0.5)

# Mark music days with red dots
music_days_data = daily_revenue_sorted[daily_revenue_sorted['has_music'] == 1]
plt.scatter(music_days_data['Date'], music_days_data['total_revenue'], color='red', alpha=0.7, label='Music Event')

plt.title('Revenue Trend with Music Events Highlighted', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Daily Revenue', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('revenue_trend_with_music.png')
plt.show()

# Figure 6: Revenue Impact by Band Compensation
daily_revenue_with_bands['Rate'] = pd.to_numeric(daily_revenue_with_bands['Rate'], errors='coerce')
rate_data = daily_revenue_with_bands.dropna(subset=['Rate'])

plt.figure(figsize=(12, 8))
plt.scatter(rate_data['Rate'], rate_data['total_revenue'], alpha=0.7)
plt.title('Band Rate vs. Daily Revenue', fontsize=16)
plt.xlabel('Band Compensation Rate', fontsize=14)
plt.ylabel('Daily Revenue', fontsize=14)
plt.grid(True, alpha=0.3)

# Add trend line
if len(rate_data) > 1:
    z = np.polyfit(rate_data['Rate'], rate_data['total_revenue'], 1)
    p = np.poly1d(z)
    plt.plot(rate_data['Rate'], p(rate_data['Rate']), "r--", alpha=0.8)
    
    # Calculate correlation
    corr = rate_data['Rate'].corr(rate_data['total_revenue'])
    plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction')

plt.tight_layout()
plt.savefig('rate_vs_revenue.png')
plt.show()

# Statistical Analysis: T-test comparing music vs no music days
music_revenue = daily_revenue[daily_revenue['has_music'] == 1]['total_revenue']
no_music_revenue = daily_revenue[daily_revenue['has_music'] == 0]['total_revenue']
t_stat, p_value = stats.ttest_ind(music_revenue, no_music_revenue, equal_var=False)
print(f"\nT-test comparing music vs. no music days:")
print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
print(f"Mean revenue with music: ${music_revenue.mean():.2f}")
print(f"Mean revenue without music: ${no_music_revenue.mean():.2f}")
print(f"Revenue difference: ${music_revenue.mean() - no_music_revenue.mean():.2f}")

# Multiple regression analysis
X = sm.add_constant(daily_revenue[['has_music', 'is_weekend']])
y = daily_revenue['total_revenue']
model = sm.OLS(y, X).fit()
print("\nRegression Analysis Results:")
print(model.summary().tables[1])  # Print just the coefficients table

# Analyze which bands perform on which days
band_days = music.groupby(['Band_ID', 'Weekday']).size().reset_index(name='performances')
top_band_days = band_days.sort_values('performances', ascending=False).head(15)
print("\nMost Common Band-Day Combinations:")
print(top_band_days)