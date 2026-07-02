import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

# Import custom preprocessing
try:
    from data_preprocessing import map_raw_df_to_features, prepare_ml_data
except ImportError:
    from src.data_preprocessing import map_raw_df_to_features, prepare_ml_data

def train_and_evaluate_models():
    print("Starting Placement Intelligence Model Training pipeline...")
    
    # 1. Load data
    data_path = "Placement_Data_Enriched.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Enriched dataset not found at {data_path}. Please run generate_data.py first.")
        
    raw_df = pd.read_csv(data_path)
    
    # 2. Extract features & preprocess
    print("Preprocessing raw data...")
    features_df = map_raw_df_to_features(raw_df)
    
    # Save the original dataframe shape and features
    X, y, scaler, feature_names = prepare_ml_data(features_df)
    
    # 3. Train-Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset split: Train shape={X_train.shape}, Test shape={X_test.shape}")
    
    # 4. Initialize classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
    }
    
    metrics_summary = {}
    best_f1 = -1
    best_model_name = None
    best_model_obj = None
    
    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        
        # Hyperparameter tuning for selected models to meet the GridSearch/tuning requirement
        if name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 150],
                'max_depth': [8, 12]
            }
            grid = GridSearchCV(clf, param_grid, cv=3, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)
            clf = grid.best_estimator_
            print(f"  Best params for Random Forest: {grid.best_params_}")
        elif name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 150],
                'max_depth': [4, 6]
            }
            grid = GridSearchCV(clf, param_grid, cv=3, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)
            clf = grid.best_estimator_
            print(f"  Best params for XGBoost: {grid.best_params_}")
        else:
            clf.fit(X_train, y_train)
            
        # Predictions
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred).tolist() # Convert to list for JSON serialization
        
        # ROC Curve coordinates
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        # Downsample ROC coordinates for smaller JSON storage size
        downsample_rate = max(1, len(fpr) // 100)
        fpr_list = fpr[::downsample_rate].tolist()
        tpr_list = tpr[::downsample_rate].tolist()
        
        metrics_summary[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(auc),
            'confusion_matrix': cm,
            'fpr': fpr_list,
            'tpr': tpr_list
        }
        
        print(f"  {name} Metrics: F1={f1:.4f}, Accuracy={acc:.4f}, ROC_AUC={auc:.4f}")
        
        # Track the best model based on F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = clf
            
    print(f"\nBest Classifier selected: {best_model_name} with F1-Score: {best_f1:.4f}")
    
    # 5. Train Expected Salary Regressor
    # We will train a Random Forest Regressor on all records to predict Expected Salary
    print("Training Expected Salary Regressor...")
    y_salary = features_df['Expected Salary'].values
    
    # Train-test split for salary regression
    X_train_sal, X_test_sal, y_train_sal, y_test_sal = train_test_split(
        X, y_salary, test_size=0.2, random_state=42
    )
    
    # Train regressor
    regressor = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    regressor.fit(X_train_sal, y_train_sal)
    
    # Evaluate regressor
    y_pred_sal = regressor.predict(X_test_sal)
    mae = np.mean(np.abs(y_test_sal - y_pred_sal))
    rmse = np.sqrt(np.mean((y_test_sal - y_pred_sal)**2))
    r2 = float(regressor.score(X_test_sal, y_test_sal))
    
    # Standard deviation of residuals for confidence intervals
    residuals = y_test_sal - y_pred_sal
    residual_std = float(np.std(residuals))
    
    salary_metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': r2,
        'residual_std': residual_std
    }
    print(f"Regressor performance: MAE={mae:.4f} LPA, RMSE={rmse:.4f} LPA, R2={r2:.4f}, Residual Std={residual_std:.4f}")
    
    # Create output directory for model assets if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # 6. Save assets
    print("Saving trained models and scalers...")
    joblib.dump(best_model_obj, "models/best_classifier.joblib")
    joblib.dump(regressor, "models/salary_regressor.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(feature_names, "models/feature_names.joblib")
    
    # Save the metrics summary to JSON for the dashboard comparison tab
    metrics_summary['best_model'] = best_model_name
    metrics_summary['salary_metrics'] = salary_metrics
    
    with open("models/model_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print("All model assets saved successfully in 'models/' folder!")

if __name__ == "__main__":
    train_and_evaluate_models()
