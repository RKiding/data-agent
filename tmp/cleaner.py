def data_cleaner(data_raw):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer
    
    # Create a copy of the input data to avoid modifying the original
    data_cleaned = data_raw.copy()
    
    # Step 1: Convert 'low' column to numeric type
    data_cleaned['low'] = pd.to_numeric(data_cleaned['low'], errors='coerce')
    
    # Step 2: Handle missing values in 'volume' and 'open' columns by imputing with the mean
    numeric_cols = ['volume', 'open']
    imputer = SimpleImputer(strategy='mean')
    data_cleaned[numeric_cols] = imputer.fit_transform(data_cleaned[numeric_cols])
    
    # Step 3: Remove duplicate rows
    data_cleaned = data_cleaned.drop_duplicates()
    
    # Step 4: Remove rows with extreme outliers in numeric columns
    numeric_cols_all = ['open', 'high', 'low', 'close', 'volume']
    
    for col in numeric_cols_all:
        if col in data_cleaned.columns:
            # Calculate Q1, Q3 and IQR
            Q1 = data_cleaned[col].quantile(0.25)
            Q3 = data_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter out outliers
            data_cleaned = data_cleaned[(data_cleaned[col] >= lower_bound) & 
                                      (data_cleaned[col] <= upper_bound)]
    
    # Step 5: Verify data types after cleaning
    # Convert all numeric columns to float64 for consistency
    for col in numeric_cols_all:
        if col in data_cleaned.columns:
            data_cleaned[col] = data_cleaned[col].astype('float64')
    
    return data_cleaned