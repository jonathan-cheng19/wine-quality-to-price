import pandas as pd

# Read the Red.csv file
df = pd.read_csv('Red.csv')

# Sort by Grape variety
df_sorted = df.sort_values(by='Grape')

# Save the sorted data back to Red.csv
df_sorted.to_csv('Red.csv', index=False)

print(f"Sorted {len(df_sorted)} wines by grape variety\n")

# Calculate price statistics for each grape variety
grape_stats = df_sorted.groupby('Grape')['Price'].agg([
    ('Count', 'count'),
    ('Min_Price', 'min'),
    ('Max_Price', 'max'),
    ('Avg_Price', 'mean'),
    ('Median_Price', 'median')
]).sort_values('Count', ascending=False)

# Format the output
grape_stats['Price_Range'] = grape_stats.apply(
    lambda x: f"${x['Min_Price']:.2f} - ${x['Max_Price']:.2f}", axis=1
)
grape_stats['Avg_Price'] = grape_stats['Avg_Price'].apply(lambda x: f"${x:.2f}")
grape_stats['Median_Price'] = grape_stats['Median_Price'].apply(lambda x: f"${x:.2f}")

# Display the results
print("=" * 100)
print(f"{'Grape Variety':<35} {'Count':>8} {'Price Range':>25} {'Average':>12} {'Median':>12}")
print("=" * 100)

for grape, row in grape_stats.iterrows():
    print(f"{grape:<35} {int(row['Count']):>8} {row['Price_Range']:>25} {row['Avg_Price']:>12} {row['Median_Price']:>12}")

print("=" * 100)
print(f"\nTotal wines: {len(df_sorted)}")
