def feature_engineer(data_raw):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    # Create a copy of the input data to avoid modifying the original
    data_engineered = data_raw.copy()
    
    # 1. Convert Data Types - already appropriate (float64 for prices, int64 for volume)
    
    # 2. Remove Constant Features - none found (all columns have sufficient unique values)
    
    # 3. High Cardinality Categorical Features - no categorical features in this dataset
    
    # 4. One-Hot Encoding - no categorical features to encode
    
    # 5. Numeric Features - create technical indicators and normalize
    # Calculate price changes
    data_engineered['price_change'] = data_engineered['close'] - data_engineered['open']
    data_engineered['high_low_spread'] = data_engineered['high'] - data_engineered['low']
    
    # Calculate percentage changes
    data_engineered['pct_change'] = (data_engineered['close'] - data_engineered['open']) / data_engineered['open']
    
    # Calculate simple moving averages
    for window in [5, 10, 20]:
        data_engineered[f'sma_{window}'] = data_engineered['close'].rolling(window=window).mean()
        data_engineered[f'ema_{window}'] = data_engineered['close'].ewm(span=window, adjust=False).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = data_engineered['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data_engineered['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    data_engineered['bb_ma'] = data_engineered['close'].rolling(20).mean()
    data_engineered['bb_upper'] = data_engineered['bb_ma'] + 2 * data_engineered['close'].rolling(20).std()
    data_engineered['bb_lower'] = data_engineered['bb_ma'] - 2 * data_engineered['close'].rolling(20).std()
    
    # Calculate volume features
    data_engineered['volume_pct_change'] = data_engineered['volume'].pct_change()
    data_engineered['volume_ma_5'] = data_engineered['volume'].rolling(5).mean()
    data_engineered['volume_ma_20'] = data_engineered['volume'].rolling(20).mean()
    
    # 6. Datetime Features - no datetime features in this dataset
    
    # 7. Target Variable - none provided
    
    # 8. Boolean Conversion - no boolean features in this dataset
    
    # 9. Custom Feature Engineering for Financial Data
    # Calculate typical price
    data_engineered['typical_price'] = (data_engineered['high'] + data_engineered['low'] + data_engineered['close']) / 3
    
    # Calculate money flow
    data_engineered['money_flow'] = data_engineered['typical_price'] * data_engineered['volume']
    
    # Calculate MACD (Moving Average Convergence Divergence)
    ema_12 = data_engineered['close'].ewm(span=12, adjust=False).mean()
    ema_26 = data_engineered['close'].ewm(span=26, adjust=False).mean()
    data_engineered['macd'] = ema_12 - ema_26
    data_engineered['macd_signal'] = data_engineered['macd'].ewm(span=9, adjust=False).mean()
    
    # 10. Final Check
    # Fill any NaN values created by rolling calculations
    data_engineered = data_engineered.fillna(method='bfill').fillna(method='ffill')
    
    # Normalize numeric features (excluding the original columns)
    original_cols = ['open', 'high', 'low', 'close', 'volume']
    numeric_cols = [col for col in data_engineered.columns if col not in original_cols]
    scaler = StandardScaler()
    data_engineered[numeric_cols] = scaler.fit_transform(data_engineered[numeric_cols])
    
    return data_engineered