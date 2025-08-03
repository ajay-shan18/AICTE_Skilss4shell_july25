import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import re

# Data cleaning function
def clean_data(data):
    # Remove the UUID and "Page" lines
    cleaned = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}Page \d+', '', data)
    # Remove extra whitespace and empty lines
    cleaned = '\n'.join([line.strip() for line in cleaned.split('\n') if line.strip()])
    return cleaned

# Process the raw text content
raw_data = """[your entire text content from above]"""
cleaned_data = clean_data(raw_data)

# Create a DataFrame
df = pd.read_csv(StringIO(cleaned_data))

# Data Exploration
print("Data Overview:")
print(df.head())
print("\nData Statistics:")
print(df.describe())

# Visualization 1: Sensor Value Distributions
plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns[:8]):  # First 8 sensors
    plt.subplot(3, 3, i+1)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.suptitle('Sensor Value Distributions', y=1.02)
plt.show()

# Visualization 2: Correlation Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Sensor Correlation Heatmap')
plt.show()

# Visualization 3: Time Series Plot for First 100 Rows
plt.figure(figsize=(15, 8))
for sensor in df.columns[:5]:  # First 5 sensors
    plt.plot(df[sensor][:100], label=sensor)
plt.xlabel('Time Steps')
plt.ylabel('Sensor Values')
plt.title('Time Series of Sensor Readings (First 100 Rows)')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 4: Box Plots
plt.figure(figsize=(15, 6))
sns.boxplot(data=df[df.columns[:8]])  # First 8 sensors
plt.title('Box Plot of Sensor Readings')
plt.xticks(rotation=45)
plt.show()

# Visualization 5: Pair Plot for first 5 sensors
sns.pairplot(df[df.columns[:5]])
plt.suptitle('Pair Plot of Sensor Relationships', y=1.02)
plt.show()

# Advanced Analysis: Outlier Detection
plt.figure(figsize=(15, 8))
for i, col in enumerate(df.columns[:6]):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x=df[col])
    plt.title(f'Outliers in {col}')
plt.tight_layout()
plt.show()

# Additional Analysis: Rolling Averages
plt.figure(figsize=(15, 6))
for sensor in df.columns[:3]:
    rolling_avg = df[sensor].rolling(window=20).mean()
    plt.plot(rolling_avg, label=f'{sensor} (20-step avg)')
plt.title('Rolling Averages of Sensor Readings')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()
