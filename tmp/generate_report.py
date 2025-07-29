def generate_report(data, eda_results, model_results, visualizations, output_file="report.html"):
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import os
    
    # Get current date for report
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Prepare visualization section
    vis_section = ""
    if visualizations:
        vis_section = f"""
        <h2>Visualizations</h2>
        <p>The analysis includes {len(visualizations)} key visualizations that help illustrate the data patterns and relationships.</p>
        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            {"".join([f'<img src="{vis}" style="max-width: 300px; height: auto; border: 1px solid #ddd;">' for vis in visualizations])}
        </div>
        """
    
    # Prepare model results section
    model_section = """
    <h2>Model Results</h2>
    <p>No model results were provided for this analysis.</p>
    """
    if model_results:
        model_section = f"""
        <h2>Model Results</h2>
        <p>Key model performance metrics:</p>
        <ul>
            <li>Accuracy: {model_results.get('accuracy', 'N/A')}</li>
            <li>Precision: {model_results.get('precision', 'N/A')}</li>
            <li>Recall: {model_results.get('recall', 'N/A')}</li>
            <li>F1 Score: {model_results.get('f1_score', 'N/A')}</li>
        </ul>
        """
    
    # Prepare EDA results section
    eda_section = ""
    if eda_results:
        eda_section = f"""
        <h2>Exploratory Data Analysis</h2>
        <p>Key findings from EDA:</p>
        <ul>
            <li>Missing values: {eda_results.get('missing_values', 'N/A')}</li>
            <li>Data types: {eda_results.get('data_types', 'N/A')}</li>
            <li>Correlation analysis: {eda_results.get('correlation', 'N/A')}</li>
            <li>Outlier detection: {eda_results.get('outliers', 'N/A')}</li>
        </ul>
        """
    
    # HTML report template
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Science Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #2980b9; margin-top: 30px; }}
            .summary-box {{ background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 20px 0; }}
            .data-overview {{ display: flex; gap: 30px; margin: 20px 0; }}
            .data-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; flex: 1; }}
            .insights {{ background-color: #e8f4fc; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .footer {{ margin-top: 40px; text-align: center; font-size: 0.9em; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Data Science Analysis Report</h1>
            <p class="footer">Generated on {current_date}</p>
            
            <h2>Executive Summary</h2>
            <div class="summary-box">
                <p>This report summarizes the findings from the analysis of a dataset containing {data.shape[0]} records and {data.shape[1]} features. 
                The exploratory data analysis revealed key patterns and characteristics in the data. 
                {"No model was trained as part of this analysis." if not model_results else "The modeling phase produced results with the metrics shown below."}</p>
            </div>
            
            <h2>Data Overview</h2>
            <div class="data-overview">
                <div class="data-card">
                    <h3>Dataset Dimensions</h3>
                    <p>Rows: {data.shape[0]}</p>
                    <p>Columns: {data.shape[1]}</p>
                </div>
                <div class="data-card">
                    <h3>Data Sample</h3>
                    {data.head().to_html()}
                </div>
            </div>
            
            {eda_section}
            
            {model_section}
            
            {vis_section}
            
            <h2>Key Insights and Recommendations</h2>
            <div class="insights">
                <h3>Main Findings</h3>
                <ul>
                    <li>The dataset appears to be well-structured with a mix of numerical and categorical features</li>
                    <li>Initial analysis shows interesting patterns that warrant further investigation</li>
                    <li>Visualizations highlight key relationships between variables</li>
                </ul>
                
                <h3>Recommendations</h3>
                <ul>
                    <li>Consider additional feature engineering to improve model performance</li>
                    <li>Address any data quality issues identified during EDA</li>
                    <li>Explore more advanced modeling techniques if predictive performance is a key objective</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>Report generated by Data Science Report Generation Agent</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save the report to file
    with open(output_file, 'w') as f:
        f.write(html_report)
    
    return output_file