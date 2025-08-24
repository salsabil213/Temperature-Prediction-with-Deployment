#!/usr/bin/env python3
"""
Health check script for Temperature Prediction Streamlit app
This script validates that all components work correctly
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def check_data_files():
    """Check if required data files exist and are readable"""
    required_files = [
        'df_model_after_Pettit.csv',
        'df_model_before_Pettit.csv'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing required file: {file}")
            return False
        
        try:
            df = pd.read_csv(file)
            print(f"✅ {file}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
            return False
    
    return True

def check_dependencies():
    """Check if all required packages are available"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'sklearn', 'openpyxl'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ Missing package: {package}")
            return False
    
    return True

def check_model_training():
    """Test model training functionality"""
    try:
        # Load data
        df = pd.read_csv('df_model_after_Pettit.csv')
        
        # Prepare features
        feature_cols = ['Barometer - mm Hg', 'Hum - %', 'Wind Speed - m/s', 
                       'Rain - mm', 'Heat Index - °C', 'Season', 'Day_Night']
        
        df_clean = df[feature_cols + ['Temp - °C']].dropna()
        X = df_clean[feature_cols]
        y = df_clean['Temp - °C']
        
        # Train small model for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
        model = RandomForestRegressor(n_estimators=5, random_state=42, max_depth=3)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Model training successful: R² = {r2:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def main():
    """Run all health checks"""
    print("🌡️ Temperature Prediction App Health Check")
    print("=" * 50)
    
    checks = [
        ("Data Files", check_data_files),
        ("Dependencies", check_dependencies),
        ("Model Training", check_model_training)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All health checks passed! App is ready for deployment.")
        sys.exit(0)
    else:
        print("❌ Some health checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()