# ----- Research Question: How does cycling infrastructure (type of road intersection) influence the number and severity of cycling incidents in Amsterdam? -----#

# Importing necessary libraries
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Running cycling dataset
cycling_df = pd.read_csv('csv tables/prediction_data_cycling.csv', low_memory=False)

# Selecting relevant columns from dataframe
selected_columns = cycling_df[[
    'JAAR_VKL', 'OTE_ID', 'MAXSNELHD', 'WDK_ID', 'WSE_ID', 'AP3_CODE']]

# Dictionary to map Dutch column names to English
column_translation = {
    'JAAR_VKL': 'Year',
    'OTE_ID': 'Type of bicycle',
    'MAXSNELHD': 'Speed limit',
    'WDK_ID': 'Road condition',
    'WSE_ID': 'Intersection type',
    'AP3_CODE': 'Accident type'
}

# Apply these renamed columns to the dataframe
selected_columns_english = selected_columns.rename(columns=column_translation)

# Dropping NaN values
selected_columns_english.dropna(inplace=True)

# Convert float columns to int on the renamed dataframe
cols_to_convert = ['Road condition', 'Intersection type', 'Speed limit']
for col in cols_to_convert:
    selected_columns_english[col] = selected_columns_english[col].astype(int)

# Save to csv
selected_columns_english.to_csv('csv tables/cleaned_prediction_data_cycling.csv', index=False)

# -----------------------------------------------------------------------------------------------------------------

# Hypothesis 1： Intersections at higher speed limits have significantly more crashes than straight roads at the same limit #

# (1) Plotting bar graph with speed limit against number of accidents-- to compare the volume of crashes across types and limits

# Rerun cleaned cycling dataframe
cycling_df = pd.read_csv('csv tables/cleaned_prediction_data_cycling.csv', low_memory=False)

# Get min and max
min_speed = cycling_df['Speed limit'].min()
max_speed = cycling_df['Speed limit'].max()
print(min_speed)
print(max_speed)

# Group and count crashes using Accident type column
crash_counts = cycling_df.groupby(['Speed limit', 'Intersection type'])[
    'Accident type'].count().reset_index(name='Crash Count')

# Mapping intersection codes into clearer categories
layout_map = {
    1: 'Straight road',
    6: 'Straight road',
    4: 'Intersection road',
    5: 'Intersection road'
}

cycling_df['Road Layout Group'] = cycling_df['Intersection type'].map(
    layout_map)

# Groupping the speed limit, mapped road layout and accident type
crash_counts = cycling_df.groupby(['Speed limit', 'Road Layout Group'])['Accident type'] \
                         .count().reset_index(name='Crash Count')

# Plot bar graph with speed limit against number of accidents
plt.figure(figsize=(12, 6))
sns.barplot(data=crash_counts, x='Speed limit',
            y='Crash Count', hue='Road Layout Group')
plt.title('Crash Counts by Speed Limit and Road Layout Type')
plt.ylabel('Number of Accidents')
plt.xlabel('Speed Limit (km/h)')
plt.legend(title='Road Layout Type')
plt.xticks(rotation=45)
plt.tight_layout()

# Save bar graph to folder
plt.savefig('visualisations/crash_counts_by_speed_limit_and_road_layout_type.png')

# (2) Run a statistical test using Poisson regression model

# Group and count again using renamed intersection types
crash_counts = cycling_df.groupby(['Speed limit', 'Intersection type'])['Accident type'] \
                         .count().reset_index(name='CrashCount')

# Mapping road layout codes into their specific names
intersection_names = {
    1: "Straight road",
    4: "3-way intersection",
    5: "4-way intersection",
    6: "Straight road w/ separated lanes",
}

# Create a cleaned name column
crash_counts['Intersection_Name'] = crash_counts['Intersection type'].map(
    intersection_names)

# Ensure speed limit is numeric
crash_counts['Speed limit'] = pd.to_numeric(crash_counts['Speed limit'])

# Fit Poisson regression using interaction formula
model = smf.glm(
    formula='CrashCount ~ C(Intersection_Name) * Q("Speed limit")',
    data=crash_counts,
    family=sm.families.Poisson()
).fit()

# Save model summary to folder
with open('data analysis/model_summary.txt', 'w') as f:
    f.write(model.summary().as_text())

# (3) Plotting line graph with speed limit against number of accidents-- to demonstrate how the intersection types behave differently as speed increases (ex. road complexity)

# Mapping intersection codes into their specific names
intersection_names = {
    4: "3-way intersection",
    5: "4-way intersection",
}

# Create a cleaned name column
crash_counts['Intersection_Name'] = crash_counts['Intersection type'].map(
    intersection_names)

# Plot line graph with spped limit versus the number of accidents
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=crash_counts,
    x='Speed limit',
    y='CrashCount',
    hue='Intersection_Name',
    marker='o'
)

plt.title('Crash Counts by Speed Limit and Intersection Type')
plt.ylabel('Number of Accidents')
plt.xlabel('Speed Limit (km/h)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(title='Intersection Type')
plt.grid(True)

# Save line graph to folder
plt.savefig('visualisations/crash_counts_by_speed_limit.png')

# -----------------------------------------------------------------------------------------------------------------

# Hypothesis 2: Crashes occurring at intersections result in more severe human injuries compared to crashes on straight roads.

# (1) Plotting seperate bar graphs with accident types against number of accidents

# Mapping road layout codes to actual meanings
layout_map = {
    1: 'Straight road',
    6: 'Straight road',
    4: 'Intersection road',
    5: 'Intersection road'
}
cycling_df['Road Layout Group'] = cycling_df['Intersection type'].map(
    layout_map)

# Mapping accident type codes to categories
accident_map = {
    "DOD": 'Deaths',
    "LET": 'Injurity',
    "UMS": 'Material damages'
}
cycling_df['Accident Category'] = cycling_df['Accident type'].map(accident_map)

# Groupping data of accident type and the mapped accident type and road layout
crash_counts = cycling_df.groupby(
    ['Accident type', 'Accident Category', 'Road Layout Group']
).size().reset_index(name='Crash Count')

# Plot seperate bar graphs of accident type versus the number of crashes
g = sns.catplot(
    data=crash_counts,
    x='Accident Category',
    y='Crash Count',
    hue='Accident type',
    col='Road Layout Group',
    kind='bar',
    height=5,
    aspect=1
)

# Setting labels and titles
g.set_axis_labels("Accident Type", "Number of Accidents")
g.set_titles("{col_name}")

# Adjust legend position
g._legend.set_title('Type of injury')
# Move legend right/left and top/down
g._legend.set_bbox_to_anchor((0.97, 0.85))
g._legend.set_frame_on(True)

# Rotate x-axis labels
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(45)

plt.tight_layout()

# Save bar graph to folder
plt.savefig('visualisations/crash_counts_by_road_and_accident_type.png')

# (2) Running a One sided (Mann-Whitney U test)
# Set a severity scale
severity_scale = {"DOD": 3, "LET": 2, "UMS": 1}
cycling_df['Severity Score'] = cycling_df['Accident type'].map(severity_scale)

# Create two arrays of severity scores
intersection_severity = cycling_df[cycling_df['Road Layout Group']
                                   == 'Intersection road']['Severity Score']
straight_severity = cycling_df[cycling_df['Road Layout Group']
                               == 'Straight road']['Severity Score']

# To test if the severity at road intersections are significantly greater
stat, p_one_sided = mannwhitneyu(
    intersection_severity,
    straight_severity,
    alternative='greater'
)

print(f"U-statistic: {stat}")
print(f"P-value (one-sided): {p_one_sided}")
