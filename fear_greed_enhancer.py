import pandas as pd

# Load the actual data
df = pd.read_csv('fear_greed_data/fear_greed_index.csv', header=None)
df.columns = ['date', 'fng_value', 'fng_classification']
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

# Reverse it (newest first -> oldest first)
df = df.iloc[::-1].reset_index(drop=True)

# Calculate 7-day moving average
df['fng_7d_ma'] = df['fng_value'].rolling(window=7, min_periods=1).mean().round(1)

# Calculate 30-day moving average  
df['fng_30d_ma'] = df['fng_value'].rolling(window=30, min_periods=1).mean().round(1)

# Calculate rate of change first
df['fng_change'] = df['fng_value'].diff().fillna(0)
df['fng_7d_change'] = df['fng_7d_ma'].diff().fillna(0)
df['fng_30d_change'] = df['fng_30d_ma'].diff().fillna(0)

# Min-max normalization to 0-1 range
def normalize_to_01(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series * 0  # All zeros if no variation
    return (series - min_val) / (max_val - min_val)

df['fng_momentum'] = normalize_to_01(df['fng_change']).round(2)
df['fng_momentum_7d'] = normalize_to_01(df['fng_7d_change']).round(2)
df['fng_momentum_30d'] = normalize_to_01(df['fng_30d_change']).round(2)

# Drop temporary columns
df = df.drop(['fng_change', 'fng_7d_change', 'fng_30d_change'], axis=1)

# Show the first 20 rows with calculations
print("\nFirst 20 rows with moving averages and normalized momentum:")
print("Date       | FNG | 7d MA | 30d MA | Mom | Mom7d | Mom30d")
print("-" * 58)

for i in range(20):
    row = df.iloc[i]
    print(f"{row['date'].strftime('%Y-%m-%d')} | {row['fng_value']:3d} | {row['fng_7d_ma']:5.1f} | {row['fng_30d_ma']:6.1f} | {row['fng_momentum']:4.2f} | {row['fng_momentum_7d']:5.2f} | {row['fng_momentum_30d']:6.2f}")

# Save the results
df.to_csv('fear_greed_data/fear_greed_index_enhanced.csv', index=False)
print(f"\nSaved to: fear_greed_data/fear_greed_index_enhanced.csv") 