def train_models(data, target_column=None, sample_size=0):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.feature_selection import SelectKBest, f_regression
    from tqdm import tqdm
    import warnings
    from lightgbm import LGBMRegressor
    import shap












    # Initialize LightGBM availability
    lgbm_available = False
    try:

        lgbm_available = True
    except ImportError:
        print("LightGBM not available. Skipping LGBMRegressor.")
    
    # Initialize SHAP availability
    shap_available = False
    try:

        shap_available = True
    except ImportError:
        print("SHAP not available. Skipping SHAP explanations.")
    
    warnings.filterwarnings('ignore')
    
    results = {
        'best_model': None,
        'model_scores': {},
        'feature_importance': {},
        'predictions': {},
        'evaluation_metrics': {}
    }
    
    # Step 1: Problem Definition - regression confirmed
    
    # Handle sample size if specified
    if sample_size > 0:
        data = data.sample(min(sample_size, len(data)), random_state=42)
    
    # Step 2: Data Splitting
    if target_column is None:
        # Default to 'close' if no target specified
        target_column = 'close'
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Step 3: Feature Selection
    selector = SelectKBest(score_func=f_regression, k=min(10, len(X.columns)))
    selector.fit(X_train, y_train)
    selected_features = X.columns[selector.get_support()]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Step 4: Algorithm Selection
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR()
    }
    
    # Add LightGBM if available
    if lgbm_available:
        models['LightGBM'] = LGBMRegressor(random_state=42)
    
    # Step 5: Hyperparameter Tuning (simplified for example)
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6]
        }
    }
    
    # Add LightGBM params if available
    if lgbm_available:
        param_grids['LightGBM'] = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 63]
        }
    
    # Step 6: Cross-Validation and Model Training
    best_score = -np.inf
    best_model_name = None
    
    for model_name, model in tqdm(models.items(), desc="Training models"):
        try:
            # Hyperparameter tuning if parameters are available
            if model_name in param_grids:
                grid_search = RandomizedSearchCV(
                    model,
                    param_grids[model_name],
                    cv=5,
                    n_iter=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=42
                )
                grid_search.fit(X_train_selected, y_train)
                best_model = grid_search.best_estimator_
            else:
                best_model = model
                best_model.fit(X_train_selected, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(
                best_model,
                X_train_selected,
                y_train,
                cv=5,
                scoring='neg_mean_squared_error'
            )
            avg_cv_score = np.mean(cv_scores)
            
            # Test set evaluation
            y_pred = best_model.predict(X_test_selected)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results['model_scores'][model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV_Score': avg_cv_score
            }
            
            results['predictions'][model_name] = y_pred
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                importance = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                importance = np.abs(best_model.coef_)
            else:
                importance = None
                
            if importance is not None:
                feature_importance = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                results['feature_importance'][model_name] = feature_importance
            
            # Track best model
            if r2 > best_score:
                best_score = r2
                best_model_name = model_name
                results['best_model'] = best_model
                
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Step 8: Feature Importance Analysis (SHAP for best model)
    if shap_available and results['best_model'] is not None and hasattr(results['best_model'], 'predict'):
        try:
            explainer = shap.Explainer(results['best_model'])
            shap_values = explainer(X_test_selected)
            results['shap_values'] = shap_values
        except Exception as e:
            print(f"SHAP explanation failed: {str(e)}")
    
    # Step 9: Model Comparison
    results['evaluation_metrics'] = pd.DataFrame.from_dict(
        results['model_scores'], 
        orient='index'
    ).sort_values('R2', ascending=False)
    
    return results