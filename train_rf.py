import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import networkx as nx


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

# Create a Research Model
# Create a directed graph
G = nx.DiGraph()

# Add nodes
nodes = {
    'RT': 'Round Trip',
    'TM': 'Ride Length\nMinutes',
    'TD': 'Time of Day',
    'DW': 'Day of Week',
    'SE': 'Season',
    'UT': 'User Type\nMember/Casual',
    'BT': 'Bike Type',
    'SL': 'Start Location',
    'EL': 'End Location'
}

# Add nodes to the graph
for node_id, label in nodes.items():
    G.add_node(node_id, label=label)

# Add edges with positive/negative relationships
edges = [
    ('TM', 'RT', '+'),
    ('TD', 'RT', '+'),
    ('DW', 'RT', '-'),
    ('SE', 'RT', '-'),
    ('UT', 'RT', '-'),
    ('BT', 'RT', '+'),
    ('SL', 'RT', '+'),
    ('EL', 'RT', '+')
]

# Add edges to the graph
G.add_edges_from([(start, end) for start, end, _ in edges])

# Create the plot
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=1, iterations=50)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                      node_size=2000, alpha=0.7)

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color='gray', 
                      arrows=True, arrowsize=20)

# Add node labels
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels)

# Add edge labels (+ or -)
edge_labels = {(start, end): sign for start, end, sign in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels)

# Remove axis
plt.axis('off')

# Add title
plt.title('Research Model: Predictors of Round Trip Probability', 
          pad=20, size=14)

plt.tight_layout()
plt.show()

