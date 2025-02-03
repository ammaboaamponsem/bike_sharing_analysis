import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Load the data
df = pd.read_csv('all_rides_df_clean.csv')
df.describe()

# Data Cleaning
print("Before cleaning:")
print(df.shape)
print("Missing values:\n", df.isnull().sum())

# Removing rows with missing values in start_station_id and end_station_id
df = df.dropna(subset=['start_station_id', 'end_station_id'])
print(df.isnull().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)
print("\nAfter cleaning:")
print(df.shape)

# Convert datetime columns
datetime_columns = ['started_at', 'ended_at', 'started_at_date', 'ended_at_date']
for col in datetime_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Handle time columns
time_columns = ['started_at_time', 'ended_at_time']
for col in time_columns:
    df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce').dt.time

# Convert numeric columns
numeric_columns = ['ride_length_seconds', 'ride_length_minutes', 
                  'start_lat', 'start_lng', 'end_lat', 'end_lng']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create the 'round_trip' variable
df['round_trip'] = (df['start_station_id'] == df['end_station_id']).astype(int)

# Remove invalid rides
df = df[df['ride_length_seconds'] > 0]
df = df[df['ride_length_seconds'] <= 86400]  # 24 hours in seconds

# Check categorical variables
print("\nUnique values in 'rideable_type':", df['rideable_type'].unique())
print("Unique values in 'member_casual':", df['member_casual'].unique())

# Create summary statistics table
summary_stats = df[numeric_columns + ['round_trip']].describe()
summary_stats = summary_stats.round(2)
print("\nSummary Statistics:")
print(summary_stats)

# Extract hour for time-based analysis
df['hour'] = df['started_at'].dt.hour
