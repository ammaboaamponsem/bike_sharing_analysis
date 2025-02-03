import networkx as nx

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
