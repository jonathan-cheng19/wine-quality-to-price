import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Read the Red.csv file
df = pd.read_csv('Red.csv')

# Clean the data
print(f"Total wines in dataset: {len(df)}")

# Clean NumberOfRatings - remove spaces and convert to numeric
df['NumberOfRatings'] = df['NumberOfRatings'].astype(str).str.replace(' ', '').replace('', '0')
df['NumberOfRatings'] = pd.to_numeric(df['NumberOfRatings'], errors='coerce').fillna(0)

# Clean Year column - extract numeric year
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')[0]
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Calculate wine age
current_year = datetime.now().year
df['Age'] = current_year - df['Year']

# Remove rows with missing critical data
df_clean = df.dropna(subset=['Price', 'Rating', 'Age'])
df_clean = df_clean[(df_clean['Age'] >= 0) & (df_clean['Age'] <= 100)]  # Filter reasonable ages

print(f"Wines with complete data: {len(df_clean)}")

# Get top grape varieties (excluding Unknown)
top_grapes = df_clean[df_clean['Grape'] != 'Unknown']['Grape'].value_counts().head(10).index.tolist()
df_plot = df_clean[df_clean['Grape'].isin(top_grapes)]

# Create a color map for grape varieties
grapes_list = df_plot['Grape'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(grapes_list)))
grape_colors = dict(zip(grapes_list, colors))

# Create figure with single large plot
fig, ax = plt.subplots(figsize=(16, 10))
fig.suptitle('Wine Price vs Age by Grape Variety', fontsize=18, fontweight='bold')

# Plot each grape variety separately for legend
for grape in grapes_list:
    grape_data = df_plot[df_plot['Grape'] == grape]
    ax.scatter(grape_data['Age'], grape_data['Price'], 
               alpha=0.6, c=[grape_colors[grape]], s=50, 
               label=grape, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Wine Age (years)', fontsize=14, fontweight='bold')
ax.set_ylabel('Price ($)', fontsize=14, fontweight='bold')
ax.set_title(f'Scatter Plot of {len(df_plot)} Wines (Top 10 Grape Varieties)', 
             fontsize=14, pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, 
          title='Grape Variety', title_fontsize=12, framealpha=0.9)

# Add trend line for all data
z = np.polyfit(df_plot['Age'], df_plot['Price'], 2)
p = np.poly1d(z)
x_trend = np.linspace(df_plot['Age'].min(), df_plot['Age'].max(), 100)
ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label='Overall Trend', alpha=0.7)

plt.tight_layout()
plt.savefig('wine_price_analysis.png', dpi=300, bbox_inches='tight')
print("\nGraph saved as 'wine_price_analysis.png'")

# Display statistics by grape variety
print("\n" + "="*80)
print("PRICE STATISTICS BY GRAPE VARIETY (Top 10)")
print("="*80)
for grape in grapes_list:
    grape_data = df_plot[df_plot['Grape'] == grape]
    print(f"\n{grape}:")
    print(f"  Count: {len(grape_data)}")
    print(f"  Avg Price: ${grape_data['Price'].mean():.2f}")
    print(f"  Price Range: ${grape_data['Price'].min():.2f} - ${grape_data['Price'].max():.2f}")
    print(f"  Avg Age: {grape_data['Age'].mean():.1f} years")

# Display correlation statistics
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)
print(f"Correlation between Price and Rating: {df_clean['Price'].corr(df_clean['Rating']):.4f}")
print(f"Correlation between Price and Age: {df_clean['Price'].corr(df_clean['Age']):.4f}")
print(f"Correlation between Rating and Age: {df_clean['Rating'].corr(df_clean['Age']):.4f}")

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Average Price: ${df_clean['Price'].mean():.2f}")
print(f"Median Price: ${df_clean['Price'].median():.2f}")
print(f"Price Range: ${df_clean['Price'].min():.2f} - ${df_clean['Price'].max():.2f}")
print(f"\nAverage Rating: {df_clean['Rating'].mean():.2f}")
print(f"Rating Range: {df_clean['Rating'].min():.1f} - {df_clean['Rating'].max():.1f}")
print(f"\nAverage Age: {df_clean['Age'].mean():.1f} years")
print(f"Age Range: {df_clean['Age'].min():.0f} - {df_clean['Age'].max():.0f} years")

plt.show()
