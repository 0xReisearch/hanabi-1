# Data Format for Hanabi-1 Model

This directory contains example data files demonstrating the required format for training and using the Hanabi-1 model. The model requires two primary data sources:

## 1. Hourly Market Data (`hourly_data_example.csv`)

This file contains hourly price data and related metrics with the following columns:

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| timestamp | ISO datetime of the hour | Datetime string | 2024-01-01 00:00:00 |
| price | Market price at end of hour | Float | 42568.23 |
| volume | Trading volume in that hour | Float | 1245.67 |
| high | Highest price during the hour | Float | 42612.45 |
| low | Lowest price during the hour | Float | 42501.78 |
| open | Price at start of hour | Float | 42550.34 |
| close | Price at end of hour (same as price) | Float | 42568.23 |
| bid | Highest bid at end of hour | Float | 42565.45 |
| ask | Lowest ask at end of hour | Float | 42571.32 |
| spread | Difference between ask and bid | Float | 5.87 |
| volatility | Realized volatility in that hour | Float | 1.23 |

**Requirements:**
- Must be hourly data with no missing hours
- Timestamps must be in ascending order
- No missing values allowed
- At least 1000 consecutive hours for reliable training

## 2. Fear & Greed Index Data (`fear_greed_index_example.csv`)

This file contains daily market sentiment data:

| Column | Description | Data Type | Example |
|--------|-------------|-----------|---------|
| date | Date of the fear & greed reading | Date string | 2024-01-01 |
| value | Fear & Greed index value (0-100) | Integer | 65 |
| classification | Textual classification | String | Greed |
| btc_dominance | BTC market dominance percentage | Float | 52.3 |
| btc_volatility | BTC volatility measure | Float | 1.45 |
| market_momentum | Market momentum score | Float | 0.87 |
| social_sentiment | Social media sentiment score | Float | 0.65 |
| market_volume | Relative market volume score | Float | 1.23 |

**Requirements:**
- Daily data (model will interpolate to hourly)
- Dates must be in ascending order
- No missing values allowed
- Should cover the same time period as hourly data

## Data Preprocessing

The model's data preprocessor will:
1. Align timestamps between hourly and fear & greed data
2. Calculate derived features (price changes, moving averages, etc.)
3. Normalize features to appropriate ranges
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
python train_model.py --hourly_data /path/to/your/hourly_data.csv --fear_greed_data /path/to/your/fear_greed_data.csv
```