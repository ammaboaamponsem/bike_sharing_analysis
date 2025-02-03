# EDA VISUALIZATIONS

# 1. Round Trip Probability by Rider Type
plt.figure(figsize=(10, 6))
rider_type_round_trip = df.groupby('member_casual')['round_trip'].mean()
rider_type_round_trip.plot(kind='bar')
plt.title('Probability of Round Trip by Rider Type')
plt.ylabel('Probability of Round Trip')
plt.xlabel('Rider Type')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 2. Round Trip Probability by Time of Day
plt.figure(figsize=(12, 6))
hourly_round_trip = df.groupby('hour')['round_trip'].mean()
hourly_round_trip.plot()
plt.title('Probability of Round Trip by Hour of Day')
plt.ylabel('Probability of Round Trip')
plt.xlabel('Hour of Day')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()

# 3. Round Trip Probability by Bike Type
plt.figure(figsize=(10, 6))
bike_type_round_trip = df.groupby('rideable_type')['round_trip'].mean()
bike_type_round_trip.plot(kind='bar')
plt.title('Probability of Round Trip by Bike Type')
plt.ylabel('Probability of Round Trip')
plt.xlabel('Bike Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Correlogram of numeric variables
numeric_df = df[numeric_columns + ['round_trip']].copy()
corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix of Numeric Variables')
plt.tight_layout()
plt.show()

# 5. Hourly Usage Patterns by Member Type
hourly_usage = df.groupby(['hour', 'member_casual'])['ride_id'].count().unstack()
hourly_usage_normalized = hourly_usage.div(hourly_usage.sum(axis=0), axis=1)

plt.figure(figsize=(12, 6))
hourly_usage_normalized.plot(kind='line', marker='o')
plt.title('Normalized Hourly Usage by Rider Type')
plt.xlabel('Hour of Day')
plt.ylabel('Proportion of Daily Rides')
plt.legend(title='Rider Type')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
