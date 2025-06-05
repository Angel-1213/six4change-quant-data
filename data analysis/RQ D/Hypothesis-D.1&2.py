## Research Question: How does cycling infrastructure, including the type of road pavement and intersection, influence the number and severity of cycling accidents in Amsterdam? ##

# Importing necessary libraries
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Running cycling dataset
cycling_df = pd.read_csv(
    r'..\..\csv tables\prediction_data.csv', low_memory=False)

# Selecting relevant columns from dataframe
selected_columns = cycling_df[[
    'JAAR_VKL', 'WVG_ID', 'WSE_ID', 'AP3_CODE']]

# Dictionary to map Dutch column names to English
column_translation = {
    'JAAR_VKL': 'Year',
    'WVG_ID': 'Pavement type',
    'WSE_ID': 'Road type',
    'AP3_CODE': 'Accident type'
}

# Apply these renamed columns to the dataframe
selected_columns_english = selected_columns.rename(columns=column_translation)

# Dropping NaN values
selected_columns_english.dropna(inplace=True)

# Convert float columns to int on the renamed dataframe
cols_to_convert = ['Road type', 'Pavement type']
for col in cols_to_convert:
    selected_columns_english[col] = selected_columns_english[col].astype(int)

# Save to csv
selected_columns_english.to_csv(r'..\csv tables.csv\cycling.csv', index=False)

# -----------------------------------------------------------------------------------------------------------------

# Hypothesis 1: There is significantly more material damage than human damage on asphalt compared to concrete road surfaces. #

# Rerun cleaned cycling dataframe
cycling_df = pd.read_csv(r'..\..\csv tables.csv\cycling.csv', low_memory=False)

# Printing unique values of the pavement type column
print(cycling_df['Pavement type'].unique())

# Mapping pavement type codes to names
pavement_map = {
    1: 'asphalt',
    2: 'asphalt',
    3: 'concrete',
}
cycling_df['Pavement Group'] = cycling_df['Pavement type'].map(
    pavement_map)

# Mapping the accident type codes to names
accident_map = {
    "DOD": 'Death',
    "LET": 'Injury',
    "UMS": 'Material damage'
}
cycling_df['Accident Group'] = cycling_df['Accident type'].map(
    accident_map)

# Group by both pavement and accident type to count crashes
crash_counts = cycling_df.groupby(['Pavement Group', 'Accident Group']) \
                         .size().reset_index(name='Crash Count')

# Normalise counts for each pavement type to get percentages
crash_counts['Total by Pavement'] = crash_counts.groupby(
    'Pavement Group')['Crash Count'].transform('sum')
crash_counts['Percentage'] = (
    crash_counts['Crash Count'] / crash_counts['Total by Pavement']) * 100

# Plot grouped bar chart
g = sns.catplot(
    data=crash_counts,
    x='Pavement Group',
    y='Percentage',
    hue='Accident Group',
    kind='bar',
    height=5,
    aspect=1.4
)

g.set_titles("Normalized cycling accident proportions by pavement type")
g.set_axis_labels("Pavement Type", "Percentage of accidents (%)")

# Update the title and layout
g.fig.suptitle("Proportion of Cycling Accidents by Pavement Type", fontsize=14)
g.fig.subplots_adjust(top=0.9)

# Adjust legend
g._legend.set_title("Accident Type")
g._legend.set_bbox_to_anchor((0.93, 0.80))
g._legend.set_frame_on(True)

# Rotate x-axis labels
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(45)

plt.tight_layout()

# Save bar graph
plt.savefig(
    r'..\..\visualisations\Proportion_of_cycling_accidents_by_pavement_type.png')

# Conducting a one-sided z-test

# Aggregate 'special asphalt' and 'asphalt' into a single category
crash_counts['Pavement Group'] = crash_counts['Pavement Group'].replace(
    {'special asphalt': 'asphalt'}
)

# Pivot to get counts by pavement group and accident type
pivot = crash_counts.pivot_table(
    index='Pavement Group',
    columns='Accident Group',
    values='Crash Count',
    aggfunc='sum'
).fillna(0)

# Calculate Human Damage (Injuries + Deaths) and Material-to-Human Ratio
pivot['Human Damage'] = pivot['Injury'] + pivot['Death']
pivot['Material_to_Human_Ratio'] = pivot['Material damage'] / \
    pivot['Human Damage']

# Print the ratios
print(
    f"Asphalt ratio (incl. special asphalt): {pivot.loc['asphalt', 'Material_to_Human_Ratio']:.2f}")
print(
    f"Concrete ratio: {pivot.loc['concrete', 'Material_to_Human_Ratio']:.2f}")

# Prepare data for z-test
count = [
    # Material damage for asphalt
    pivot.loc['asphalt', 'Material damage'],
    pivot.loc['concrete', 'Material damage']  # Material damage for concrete
]

nobs = [
    pivot.loc['asphalt', ['Material damage', 'Injury', 'Death']
              ].sum(),  # Total asphalt accidents
    # Total concrete accidents
    pivot.loc['concrete', ['Material damage', 'Injury', 'Death']].sum()
]

z_stat, p_val = proportions_ztest(count, nobs, alternative='larger')
print(f"Z-statistic: {z_stat:.4f}, P-value: {p_val:.4f}")

# Calculate CDF-based p-value manually
manual_p_val = 1 - norm.cdf(z_stat)
print(f"Manual P-value using CDF: {manual_p_val:.4f}")

# --------------------------------------------------------------------------------------------

# Hypothesis 2: The proportion of severe accidents is higher at intersections than on straight roads. #

# Mapping road type codes to names
road_map = {
    1: "Straight road",
    4: "Intersection with 3 branches",
    5: "Intersection with 4 branches",
}

cycling_df['Road Group'] = cycling_df['Road type'].map(
    road_map)

# Mapping accident type codes to names
accident_map = {
    "DOD": 'Death',
    "LET": 'Injury',
    "UMS": 'Material damage'
}
cycling_df['Accident Group'] = cycling_df['Accident type'].map(
    accident_map)

# Group by road type and accident type and then count its occurrences
counts = cycling_df.groupby(
    ['Road Group', 'Accident Group']).size().unstack(fill_value=0)

# Calculating the severity (Injury + Death) and total accidents per road type
counts['Severe'] = counts['Injury'] + counts['Death']
counts['Total'] = counts.sum(axis=1)  # Sum of all accident types
counts['Severity Ratio'] = counts['Severe'] / counts['Total']

print(counts[['Severe', 'Total', 'Severity Ratio']])

# Calculate the proportions
counts = cycling_df.groupby(
    ['Road Group', 'Accident Group']).size().unstack(fill_value=0)
counts['Total'] = counts.sum(axis=1)
counts_normalized = counts[['Injury', 'Death', 'Material damage']].div(
    counts['Total'], axis=0) * 100  # Convert to %

# Plot stacked bar chart
# Define deep to seaborn color plalette
deep = sns.color_palette("deep").as_hex()
ax = counts_normalized.plot(
    kind='bar',
    stacked=True,
    figsize=(10, 6),
    color=deep
)

# Titles and labels
plt.title("Severity of Accidents by Road Type", fontsize=14)
plt.xlabel("Road Type")
plt.ylabel("Proportion of Accidents (%)")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Accident Type', bbox_to_anchor=(1.05, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.subplots_adjust(right=0.7)

# Annotate bars with percentages (in whole number)
for bar_index, (idx, row) in enumerate(counts_normalized.iterrows()):
    cumulative_height = 0
    for col in counts_normalized.columns:
        height = row[col]
        if height > 0.01:
            if col != 'Death' and height > 0.01:
                ax.text(
                    bar_index,
                    cumulative_height + height / 2,
                    f"{int(round(height))}%",
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='white'
                )
        cumulative_height += height

plt.tight_layout()

# Save bar graph
plt.savefig(r'..\..\visualisations\Severity_of_accidents_by_road_type.png')
