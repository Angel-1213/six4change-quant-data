from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load and clean data
file_path = Path(
    r"C:\Users\ThinkBook\Desktop\netherlands\six4change-quant-data\csv tables\mobility_by_region.csv")
data = pd.read_csv(file_path)

# 2. Filter and clean data
bicycle_data = data[(data['Margins'] == 'Value')].copy()

# Convert European number formats
numeric_cols = ['Rides per person per day (average) (number)',
                'Distance travelled per trip (average) (km)',
                'Travel time per trip (Minutes)']
for col in numeric_cols:
    bicycle_data[col] = bicycle_data[col].astype(
        str).str.replace(',', '.').astype(float)

# Change row name
bicycle_data['Travel motifs'] = bicycle_data['Travel motifs'].replace(
    'Shopping, shopping',  # Old value
    'Shopping'             # New value
)

# Filter data
bicycle_data = bicycle_data[
    (bicycle_data['Age'] != 'Total') &
    (bicycle_data['Travel motifs'] != 'Total')
].dropna(subset=['Age'])

# Drop columns
bicycle_data = bicycle_data.drop(
    ['Gender', 'Modes', 'Margins', 'Periods'], axis=1)

# 3. Save cleaned CSV
csv_output_folder = Path(
    r"C:\Users\ThinkBook\Desktop\Quantitative data analysis\csv tables")
csv_output_folder.mkdir(parents=True, exist_ok=True)
cleaned_csv_path = csv_output_folder / "cleaned_mobility_data_by_age.csv"
bicycle_data.to_csv(cleaned_csv_path, index=False)
print(f"CSV saved to: {cleaned_csv_path}")

# 4. Create and save visualization 1
viz_output_folder = Path(
    r"C:\Users\ThinkBook\Desktop\Quantitative data analysis\visualisations")
viz_output_folder.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=bicycle_data,
    x='Travel motifs',
    y='Travel time per trip (Minutes)',
    hue='Age',
    ci=None,
    palette='viridis',
    order=['To and from work', 'Shopping',  # Updated from 'Shopping, shopping'
           'Education, course, childcare', 'Leisure', 'Other travel motives']
)
plt.title("Cycling Duration by Travel motives and Age group (2023)")
plt.ylabel("Duration (minutes)")
plt.xlabel("Travel motives")
plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()

# Save figure
viz_path = viz_output_folder / "cycling_duration_by_age_and_motif.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Visualization saved to: {viz_path}")
plt.show()

# 5. Create and save visualization 2
viz_output_folder = Path(
    r"C:\Users\ThinkBook\Desktop\Quantitative data analysis\visualisations")
viz_output_folder.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=bicycle_data,
    x='Travel motifs',
    y='Distance travelled per trip (average) (km)',
    ci=None,
    palette='viridis',
    order=['To and from work', 'Shopping',  # Updated from 'Shopping, shopping'
           'Education, course, childcare', 'Leisure', 'Other travel motives']
)
plt.title("Cycling Distance by Travel Motives and age group (2023)")
plt.ylabel("Cycling distance (km)")
plt.xlabel("Travel motives")
plt.xticks(rotation=45)
plt.tight_layout()

# Save figure
viz_path = viz_output_folder / "cycling_distance_by_motif.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Visualization saved to: {viz_path}")
plt.show()
