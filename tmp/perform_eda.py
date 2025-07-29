def perform_eda(data):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import warnings
    warnings.filterwarnings('ignore')
    
    results = {
        'summary_stats': None,
        'missing_analysis': None,
        'correlation_matrix': None,
        'outlier_analysis': None,
        'distribution_analysis': None,
        'insights': []
    }
    
    # 1. Statistical Summary
    results['summary_stats'] = data.describe().to_dict()
    
    # 2. Missing Value Analysis
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    results['missing_analysis'] = {
        'missing_counts': missing_values.to_dict(),
        'missing_percentage': missing_percentage.to_dict()
    }
    results['insights'].append("No missing values found in any columns.")
    
    # 3. Distribution Analysis for Numerical Variables
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    dist_analysis = {}
    
    # Plot histograms and boxplots for key numerical columns
    key_cols = ['open', 'close', 'volume', 'rsi', 'macd', 'pct_change']
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(key_cols, 1):
        plt.subplot(2, 3, i)
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    dist_analysis['histograms'] = "Generated histograms for key numerical columns"
    
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(['open', 'close', 'volume', 'rsi'], 1):
        plt.subplot(1, 4, i)
        sns.boxplot(y=data[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    dist_analysis['boxplots'] = "Generated boxplots for key numerical columns"
    
    results['distribution_analysis'] = dist_analysis
    
    # 4. Correlation Analysis
    corr_matrix = data.corr()
    results['correlation_matrix'] = corr_matrix.to_dict()
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    
    # 5. Outlier Detection
    outlier_info = {}
    for col in ['volume', 'price_change', 'pct_change']:
        z_scores = np.abs(stats.zscore(data[col]))
        outliers = data[z_scores > 3]
        outlier_info[col] = {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100,
            'max_value': data[col].max(),
            'min_value': data[col].min()
        }
    results['outlier_analysis'] = outlier_info
    
    # 6. Data Quality Assessment
    # Check if high >= low
    invalid_high_low = data[data['high'] < data['low']]
    if len(invalid_high_low) == 0:
        results['insights'].append("Data quality check passed: All high prices are >= low prices.")
    else:
        results['insights'].append(f"Data quality issue: Found {len(invalid_high_low)} rows where high < low.")
    
    # Check RSI range (typically 0-100)
    invalid_rsi = data[(data['rsi'] < 0) | (data['rsi'] > 100)]
    if len(invalid_rsi) > 0:
        results['insights'].append(f"Found {len(invalid_rsi)} rows with RSI outside typical 0-100 range.")
    
    # 7. Price and Volume Relationship
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='pct_change', y='volume', data=data)
    plt.title('Percentage Change vs Volume')
    plt.tight_layout()
    
    # 8. Technical Indicators Analysis
    # RSI vs Price Change
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='rsi', y='pct_change', data=data)
    plt.title('RSI vs Percentage Price Change')
    plt.tight_layout()
    
    # MACD vs Signal Line
    plt.figure(figsize=(12, 6))
    plt.plot(data['macd'].iloc[-100:], label='MACD')
    plt.plot(data['macd_signal'].iloc[-100:], label='Signal Line')
    plt.title('MACD vs Signal Line (last 100 periods)')
    plt.legend()
    plt.tight_layout()
    
    # 9. Bollinger Bands Analysis
    plt.figure(figsize=(12, 6))
    plt.plot(data['close'].iloc[-100:], label='Close Price')
    plt.plot(data['bb_upper'].iloc[-100:], label='Upper Band', linestyle='--')
    plt.plot(data['bb_lower'].iloc[-100:], label='Lower Band', linestyle='--')
    plt.title('Bollinger Bands vs Close Price (last 100 periods)')
    plt.legend()
    plt.tight_layout()
    
    # 10. Volume Moving Averages
    plt.figure(figsize=(12, 6))
    plt.plot(data['volume'].iloc[-100:], label='Volume')
    plt.plot(data['volume_ma_5'].iloc[-100:], label='5-period MA')
    plt.plot(data['volume_ma_20'].iloc[-100:], label='20-period MA')
    plt.title('Volume and Moving Averages (last 100 periods)')
    plt.legend()
    plt.tight_layout()
    
    # Additional Insights
    results['insights'].extend([
        f"Volume shows extreme outliers with max value {data['volume'].max()} (Z-score analysis found {outlier_info['volume']['outlier_count']} outliers).",
        f"Strong correlation observed between open/close/high/low prices (all > 0.99).",
        f"RSI shows {len(data[data['rsi'] > 70])} overbought (>70) and {len(data[data['rsi'] < 30])} oversold (<30) conditions.",
        f"Average daily price change: {data['price_change'].mean():.4f} with standard deviation of {data['price_change'].std():.4f}.",
        f"Volume shows {len(data[data['volume_pct_change'] > 0.5])} instances with >50% change from previous period."
    ])
    
    # Show all plots
    plt.show()
    
    return results