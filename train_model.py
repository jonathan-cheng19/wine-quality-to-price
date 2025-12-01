import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading Red.csv...")
df = pd.read_csv('Red.csv')

# Clean the data
print("Cleaning data...")
df['NumberOfRatings'] = df['NumberOfRatings'].astype(str).str.replace(' ', '').replace('', '0')
df['NumberOfRatings'] = pd.to_numeric(df['NumberOfRatings'], errors='coerce').fillna(0)

# Clean Year column
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')[0]
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Calculate age
from datetime import datetime
current_year = datetime.now().year
df['Age'] = current_year - df['Year']

# Remove outliers and clean data
df_clean = df.dropna(subset=['Price', 'Rating', 'Age', 'Country', 'Region', 'Winery', 'Grape'])
df_clean = df_clean[(df_clean['Age'] >= 0) & (df_clean['Age'] <= 100)]
df_clean = df_clean[df_clean['Price'] > 0]

print(f"Clean dataset size: {len(df_clean)} wines")

# Prepare features
print("\nEncoding categorical features...")
label_encoders = {}
categorical_features = ['Country', 'Region', 'Winery', 'Grape']

for feature in categorical_features:
    le = LabelEncoder()
    df_clean[f'{feature}_encoded'] = le.fit_transform(df_clean[feature].astype(str))
    label_encoders[feature] = le

# Features for the model
feature_columns = ['Rating', 'NumberOfRatings', 'Age', 
                   'Country_encoded', 'Region_encoded', 'Winery_encoded', 'Grape_encoded']

X = df_clean[feature_columns]
y = df_clean['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
print("\nTraining Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Train Gradient Boosting model
print("Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# Evaluate models
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

rf_score = rf_model.score(X_test, y_test)
rf_predictions = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_mae = mean_absolute_error(y_test, rf_predictions)

gb_score = gb_model.score(X_test, y_test)
gb_predictions = gb_model.predict(X_test)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
gb_mae = mean_absolute_error(y_test, gb_predictions)

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)
print("\nRandom Forest:")
print(f"  R² Score: {rf_score:.4f}")
print(f"  RMSE: ${rf_rmse:.2f}")
print(f"  MAE: ${rf_mae:.2f}")

print("\nGradient Boosting:")
print(f"  R² Score: {gb_score:.4f}")
print(f"  RMSE: ${gb_rmse:.2f}")
print(f"  MAE: ${gb_mae:.2f}")

# Use the better performing model
if rf_score > gb_score:
    final_model = rf_model
    print("\nUsing Random Forest as final model")
else:
    final_model = gb_model
    print("\nUsing Gradient Boosting as final model")

# Save the model and encoders
print("\nSaving model and encoders...")
with open('wine_price_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# Save unique values for dropdowns
dropdown_data = {
    'grapes': sorted(df_clean['Grape'].unique().tolist()),
    'countries': sorted(df_clean['Country'].unique().tolist()),
    'regions': sorted(df_clean['Region'].unique().tolist()),
    'wineries': sorted(df_clean['Winery'].unique().tolist()),
}

with open('dropdown_data.pkl', 'wb') as f:
    pickle.dump(dropdown_data, f)

# Save statistics for better predictions with unknown values
model_stats = {
    'mean_price': float(df_clean['Price'].mean()),
    'median_price': float(df_clean['Price'].median()),
    'price_by_grape': df_clean.groupby('Grape')['Price'].mean().to_dict(),
    'price_by_country': df_clean.groupby('Country')['Price'].mean().to_dict(),
    'price_by_rating': df_clean.groupby('Rating')['Price'].mean().to_dict(),
}

with open('model_stats.pkl', 'wb') as f:
    pickle.dump(model_stats, f)

print("\nModel training complete!")
print("Files saved:")
print("  - wine_price_model.pkl")
print("  - label_encoders.pkl")
print("  - dropdown_data.pkl")
print("  - model_stats.pkl")

# Test predictions with sample data
print("\n" + "="*70)
print("TESTING MODEL WITH SAMPLE PREDICTIONS")
print("="*70)
sample_wines = df_clean.sample(5)
for idx, row in sample_wines.iterrows():
    features = [[row['Rating'], row['NumberOfRatings'], row['Age'],
                 row['Country_encoded'], row['Region_encoded'], 
                 row['Winery_encoded'], row['Grape_encoded']]]
    predicted = final_model.predict(features)[0]
    actual = row['Price']
    error = abs(predicted - actual)
    print(f"\nWine: {row['Name']}")
    print(f"  Actual Price: ${actual:.2f}")
    print(f"  Predicted Price: ${predicted:.2f}")
    print(f"  Error: ${error:.2f} ({error/actual*100:.1f}%)")
