import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# MODELING SECTION

# Prepare the data for modeling
# Select features and target
features = ['rideable_type', 'day_of_week', 'start_station_name', 'end_station_name', 'season', 'part_of_day', 'member_casual', 'month', 'ride_length_seconds', 'ride_length_minutes', 'ride_length_category']
target = 'round_trip'

def prepare_data(df, features, target):
    """Prepare data for modeling"""
    # Create a copy of the dataframe with only the columns we need
    model_df = df[features + [target]].copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['rideable_type', 'start_station_name', 'end_station_name',
                         'season', 'part_of_day', 'member_casual', 'month', 
                         'ride_length_category']
    
    for column in categorical_columns:
        model_df[column] = le.fit_transform(model_df[column])
    
    # Split the data
    X = model_df[features]
    y = model_df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate_rf(X_train, X_test, y_train, y_test, features):
    """Train and evaluate Random Forest model"""
    # Create and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    
    # Get feature importance
    rf_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return rf_model, rf_pred, rf_importance

def train_and_evaluate_xgb(X_train, X_test, y_train, y_test, features):
    """Train and evaluate XGBoost model"""
    # Create and train the XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Make predictions
    xgb_pred = xgb_model.predict(X_test)
    
    # Get feature importance
    xgb_importance = pd.DataFrame({
        'feature': features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return xgb_model, xgb_pred, xgb_importance

def plot_feature_importance_comparison(rf_importance, xgb_importance):
    """Plot feature importance comparison"""
    plt.figure(figsize=(15, 8))
    
    # Plot Random Forest importance
    plt.subplot(1, 2, 1)
    sns.barplot(x='importance', y='feature', data=rf_importance)
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    
    # Plot XGBoost importance
    plt.subplot(1, 2, 2)
    sns.barplot(x='importance', y='feature', data=xgb_importance)
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.show()

def print_model_comparison(y_test, rf_pred, xgb_pred):
    """Print comparison of model performance"""
    print("Random Forest Model Performance:")
    print("-" * 40)
    print("Classification Report:")
    print(classification_report(y_test, rf_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, rf_pred))
    
    print("\nXGBoost Model Performance:")
    print("-" * 40)
    print("Classification Report:")
    print(classification_report(y_test, xgb_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, xgb_pred))

# Main execution
def main(df):
    # Define features and target
    features = ['rideable_type', 'day_of_week', 'start_station_name', 
                'end_station_name', 'season', 'part_of_day', 'member_casual',
                'month', 'ride_length_seconds', 'ride_length_minutes',
                'ride_length_category']
    target = 'round_trip'
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, features, target)
    
    # Train and evaluate Random Forest
    rf_model, rf_pred, rf_importance = train_and_evaluate_rf(
        X_train, X_test, y_train, y_test, features
    )
    
    # Train and evaluate XGBoost
    xgb_model, xgb_pred, xgb_importance = train_and_evaluate_xgb(
        X_train, X_test, y_train, y_test, features
    )
    
    # Print model comparison
    print_model_comparison(y_test, rf_pred, xgb_pred)
    
    # Plot feature importance comparison
    plot_feature_importance_comparison(rf_importance, xgb_importance)
    
    return rf_model, xgb_model

# Execute the analysis
rf_model, xgb_model = main(df)

plt.close('all')
