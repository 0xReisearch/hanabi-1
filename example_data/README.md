# Data Format for Hanabi-1 Model

This directory contains example data files demonstrating the required format for training and using the Hanabi-1 model. The model requires two primary data sources:

## 1. Hourly Market Data (`hourly_data_example.csv`)

This file contains hourly price data and related metrics with the following columns:

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| Timestamp | Unix timestamp in milliseconds since epoch | Integer | 1704067200000 |
| Price | Market price at end of hour | Float | 42568.23 |
| Volume | Trading volume in that hour | Float | 1245.67 |
| High | Highest price during the hour | Float | 42612.45 |
| Low | Lowest price during the hour | Float | 42501.78 |
| Open | Price at start of hour | Float | 42550.34 |
| Close | Price at end of hour (same as price) | Float | 42568.23 |

**Requirements:**
- Must be hourly data with no missing hours
- Timestamps must be in milliseconds since epoch (Unix timestamp * 1000)
- Timestamps must be in ascending order
- No missing values allowed
- At least 12 consecutive hours for prediction (more for training)
- All column names must be capitalized as shown above

## 2. Fear & Greed Index Data (`fear_greed_index_example.csv`)

This file contains enhanced daily market sentiment data:

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| date | Date of the fear & greed reading | Date string | 2024-01-01 |
| fng_value | Fear & Greed index value (0-100) | Integer | 65 |
| fng_classification | Textual classification | String | Greed |
| fng_7d_ma | 7-day moving average of FNG index | Float | 62.5 |
| fng_30d_ma | 30-day moving average of FNG index | Float | 58.2 |
| fng_momentum | Current momentum metric | Float | 0.87 |
| fng_momentum_7d | 7-day momentum metric | Float | 0.65 |
| fng_momentum_30d | 30-day momentum metric | Float | 0.42 |

**Requirements:**
- Daily data (model will interpolate to hourly)
- Dates must be in ascending order
- No missing values allowed
- Should cover the same time period as hourly data
- Classification values must be one of: "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
- All column names must be lowercase as shown above

## Data Preprocessing

The model's data preprocessor will:
1. Align timestamps between hourly and fear & greed data
2. Calculate derived features:
   - Price range (high - low)
   - Volatility ((high - low) / low)
   - Price change percent ((close - open) / open)
   - Volume change (percent change)
   - Returns (close price percent change)
   - Moving averages (3h, 6h, 12h)
   - RSI indicators (6h, 12h)
   - Spread estimate (from volume)
3. Convert fear & greed classification to one-hot encoding
4. Create target variables (direction, volatility, price change, spread)
5. Split data into training and validation sets

For full details on data preprocessing, see the `data_preprocessor.py` file in the main directory.

## Using Your Own Data

To use your own data:
1. Format your hourly market data to match `hourly_data_example.csv`
2. Format your fear & greed data to match `fear_greed_index_example.csv`
3. Place them in the appropriate locations or specify their paths with command line arguments

Example usage:
```bash
python predict.py \
    --model_path financial_model_w12_h1_robust.pt \
    --window_size 12 \
    --hourly_data /path/to/your/hourly_data.csv \
    --fear_greed_data /path/to/your/fear_greed_data.csv
```