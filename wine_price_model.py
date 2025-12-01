import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
print("Loading Red.csv...")
df = pd.read_csv('Red.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Prepare the data
print("\n" + "="*80)
print("PREPARING DATA FOR MODELING")
print("="*80)

# Create a copy for modeling
df_model = df.copy()

# Clean the Year column - convert non-numeric years to NaN
df_model['Year'] = pd.to_numeric(df_model['Year'], errors='coerce')

# Handle missing values
df_model = df_model.dropna(subset=['Price', 'Rating', 'Year'])

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Country', 'Region', 'Winery', 'Grape']

for col in categorical_columns:
    if col in df_model.columns:
        le = LabelEncoder()
        df_model[col + '_encoded'] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} unique values")

# Select features for the model
feature_columns = ['Rating', 'NumberOfRatings', 'Year', 
                   'Country_encoded', 'Region_encoded', 'Winery_encoded', 'Grape_encoded']

# Remove any rows with missing values in feature columns
df_model = df_model.dropna(subset=feature_columns + ['Price'])

X = df_model[feature_columns]
y = df_model['Price']

print(f"\nFinal dataset size: {len(df_model)} wines")
print(f"Features: {feature_columns}")
print(f"Target variable: Price")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    results[name] = {
        'model': model,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'predictions': y_pred_test
    }
    
    print(f"  Training MAE: ${train_mae:.2f}")
    print(f"  Test MAE: ${test_mae:.2f}")
    print(f"  Training RMSE: ${train_rmse:.2f}")
    print(f"  Test RMSE: ${test_rmse:.2f}")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")

# Summary comparison
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print(f"{'Model':<25} {'Test MAE':>12} {'Test RMSE':>12} {'Test R²':>10}")
print("-"*80)

for name, metrics in results.items():
    print(f"{name:<25} ${metrics['test_mae']:>11.2f} ${metrics['test_rmse']:>11.2f} {metrics['test_r2']:>9.4f}")

# Find best model
best_model_name = min(results.items(), key=lambda x: x[1]['test_mae'])[0]
print(f"\nBest model by MAE: {best_model_name}")
print(f"This model predicts wine prices with an average error of ${results[best_model_name]['test_mae']:.2f}")

# Feature importance for Random Forest
if 'Random Forest' in results:
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("="*80)
    
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.to_string(index=False))

# Sample predictions
print("\n" + "="*80)
print("SAMPLE PREDICTIONS (Using Best Model)")
print("="*80)

best_model = results[best_model_name]['model']
sample_indices = np.random.choice(len(X_test), 10, replace=False)

if 'Linear' in best_model_name or 'Ridge' in best_model_name or 'Lasso' in best_model_name:
    sample_predictions = best_model.predict(X_test_scaled.iloc[sample_indices] if hasattr(X_test_scaled, 'iloc') else X_test_scaled[sample_indices])
else:
    sample_predictions = best_model.predict(X_test.iloc[sample_indices])

sample_actual = y_test.iloc[sample_indices]
sample_wines = df_model.loc[y_test.iloc[sample_indices].index]

print(f"{'Wine Name':<50} {'Actual':>10} {'Predicted':>10} {'Error':>10}")
print("-"*80)

for i, (idx, pred) in enumerate(zip(sample_indices, sample_predictions)):
    actual = sample_actual.iloc[i]
    wine_name = sample_wines.iloc[i]['Name'][:47]
    error = abs(actual - pred)
    print(f"{wine_name:<50} ${actual:>9.2f} ${pred:>9.2f} ${error:>9.2f}")

print("\n" + "="*80)
print(f"Model training complete! Best model: {best_model_name}")
print("="*80)
