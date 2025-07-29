def generate_visualizations(data, analysis_results=None, output_dir="visualizations"):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from matplotlib.backends.backend_pdf import PdfPages
    import warnings
    warnings.filterwarnings('ignore')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    generated_plots = []
    
    # Helper function to save plots
    def save_plot(fig, name):
        filename = os.path.join(output_dir, f"{name}.png")
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)
        generated_plots.append(filename)
    
    # 1. Price Trends Over Time
    fig, ax = plt.subplots(figsize=(15, 7))
    data[['open', 'high', 'low', 'close']].plot(ax=ax)
    ax.set_title('Price Trends Over Time', fontsize=16)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    save_plot(fig, 'price_trends')
    
    # 2. Distribution Plots for Numerical Variables
    num_vars = ['volume', 'price_change', 'pct_change']
    for var in num_vars:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data[var], kde=True, ax=ax)
        ax.set_title(f'Distribution of {var}', fontsize=14)
        save_plot(fig, f'distribution_{var}')
    
    # 3. Count Plots for Categorical Variables (if any)
    # Create a price direction categorical variable
    data['price_direction'] = np.where(data['price_change'] >= 0, 'Up', 'Down')
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='price_direction', data=data, ax=ax)
    ax.set_title('Price Direction Count', fontsize=14)
    save_plot(fig, 'price_direction_count')
    
    # 4. Correlation Heatmap
    corr_cols = ['open', 'close', 'volume', 'rsi', 'price_change', 'pct_change']
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(data[corr_cols].corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap', fontsize=16)
    save_plot(fig, 'correlation_heatmap')
    
    # 5. Box Plots for Outlier Detection
    box_vars = ['volume', 'price_change', 'pct_change']
    for var in box_vars:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(y=data[var], ax=ax)
        ax.set_title(f'Box Plot of {var}', fontsize=14)
        save_plot(fig, f'boxplot_{var}')
    
    # 6. Scatter Plots for Relationships
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='volume', y='price_change', data=data, alpha=0.5, ax=ax)
    ax.set_title('Volume vs Price Change', fontsize=14)
    save_plot(fig, 'scatter_volume_price_change')
    
    # 7. Time Series Plots for Technical Indicators
    indicators = ['sma_5', 'ema_5', 'rsi', 'macd']
    for indicator in indicators:
        fig, ax = plt.subplots(figsize=(15, 5))
        data[indicator].plot(ax=ax)
        ax.set_title(f'{indicator.upper()} Over Time', fontsize=14)
        save_plot(fig, f'timeseries_{indicator}')
    
    # 8. Feature Importance Plots (if model results available)
    if analysis_results and 'feature_importance' in analysis_results:
        fig, ax = plt.subplots(figsize=(12, 8))
        importance = analysis_results['feature_importance']
        importance.plot(kind='barh', ax=ax)
        ax.set_title('Feature Importance', fontsize=16)
        save_plot(fig, 'feature_importance')
    
    # 12. Bollinger Bands Visualization
    fig, ax = plt.subplots(figsize=(15, 7))
    data[['close', 'bb_upper', 'bb_lower']].plot(ax=ax)
    ax.set_title('Bollinger Bands', fontsize=16)
    ax.set_ylabel('Price', fontsize=12)
    save_plot(fig, 'bollinger_bands')
    
    # 13. Volume Analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    data['close'].plot(ax=ax1)
    ax1.set_title('Price with Volume', fontsize=16)
    data['volume'].plot(ax=ax2)
    ax2.set_title('Volume', fontsize=14)
    plt.tight_layout()
    save_plot(fig, 'price_volume_analysis')
    
    # 14. Money Flow Index (MFI) Visualization
    fig, ax = plt.subplots(figsize=(15, 5))
    data['money_flow'].plot(ax=ax)
    ax.set_title('Money Flow Index', fontsize=16)
    save_plot(fig, 'money_flow_index')
    
    # 15. MACD and Signal Line Plot
    fig, ax = plt.subplots(figsize=(15, 5))
    data[['macd', 'macd_signal']].plot(ax=ax)
    ax.set_title('MACD and Signal Line', fontsize=16)
    save_plot(fig, 'macd_signal')
    
    # Create a PDF with all plots
    pdf_path = os.path.join(output_dir, 'all_visualizations.pdf')
    with PdfPages(pdf_path) as pdf:
        for plot in generated_plots:
            fig = plt.figure()
            img = plt.imread(plot)
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    generated_plots.append(pdf_path)
    
    return {
        'message': f'Successfully generated {len(generated_plots)-1} visualizations',
        'plot_files': generated_plots,
        'output_directory': output_dir
    }