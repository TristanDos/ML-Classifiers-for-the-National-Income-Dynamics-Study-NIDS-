import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('IHME-GBD_2019_DATA-b7ba21db-1.csv')

# Convert 'year' column to numeric
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Filter for a specific measure (e.g., 'Prevalence')
measure = 'Prevalence'
df_measure = df[df['measure'] == measure]

# Group the data by location and year, and calculate the mean 'val' for each group
grouped = df_measure.groupby(['location', 'year'])['val'].mean().reset_index()

# Pivot the data to create a wide format
pivoted = grouped.pivot(index='year', columns='location', values='val')

# Plot the data
pivoted.plot(figsize=(12, 6))
plt.title(f'{measure} of Depressive Disorders by Location', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Location', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('plot.svg')