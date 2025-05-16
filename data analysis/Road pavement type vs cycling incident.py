from sklearn.cluster import DBSCAN
from shapely.geometry import Point
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np


# Read dataset in csv
df = pd.read_csv(
    r'C:\Users\ThinkBook\Desktop\netherlands\six4change-quant-data\data analysis', low_memory=False)

# Drop unneeded columns
columns_to_drop = ['Unnamed: 0', 'VKL_NUMMER', 'REGNUMMER', 'PVOPGEM', 'DATUM_VKL', 'DAG_CODE', 'MND_NUMMER', 'MNE_CODE',
                   'AOL_ID', 'WDK_AN', 'ZAD_ID', 'WGD_CODE_1', 'WGD_CODE_2', 'BZD_VM_AN', 'BZD_ID_IF3', 'BZD_IF_AN', 'BZD_ID_TA1',
                   'BZD_ID_TA2', 'BZD_ID_TA3', 'JTE_ID', 'HECTOMETER', 'WSE_ID', 'WSE_AN', 'WVK_ID', 'WVG_ID', 'TIJDSTIP', 'DDL_ID',
                   'AP3_CODE', 'AP4_CODE', 'AP5_CODE', 'UUR', 'ANTL_PTJ', 'NIVEAUKOP', 'WVL_ID', 'pedestrians', 'DISTRNAAM', 'ANTL_SLA', 'ANTL_DOD', 'GME_NAAM', 'PVE_CODE', 'GME_NAAM',
                   'Date', 'IND_ALC', 'DAGTYPE', 'ANTL_GZH', 'ANTL_SEH', 'BZD_ID_IF1', 'BZD_ID_IF2',
                   'OTE_ID', 'ANTL_GOV', 'GME_ID', 'ANTL_TDT', 'BZD_ID_IF3', 'BZD_ID_TA3', 'WTP_NAAM', 'BSD_NAAM', 'HUISNUMMER', 'FK_VELD5', 'PLT_NAAM',
                   'KDD_NAAM', 'DISTRCODE', 'DIENSTCODE', 'DIENSTNAAM', 'X_COORD', 'Y_COORD', 'WEEKNR', 'geometry', 'index_right',
                   'BZD_TA_AN', 'LGD_ID', 'MAXSNELHD', 'WDK_ID', 'bicycles', 'WTM_NAAM', 'Fatalities', 'BZD_ID_VM1', 'BZD_ID_VM2', 'BZD_ID_VM3']

df.drop(columns=columns_to_drop, axis=1, inplace=True)

print(df.head())

# Save cleaned data to dataset folder
csv_output_folder = Path(
    r"C:\Users\ThinkBook\Desktop\netherlands\csv tables")
csv_output_folder.mkdir(parents=True, exist_ok=True)
cleaned_csv_path = csv_output_folder / "cleaned_cycling_safety_netherlands.csv"
df.to_csv(cleaned_csv_path, index=False)

# ---------------------------------------------------------------------------#

## Finding out the district in Netherlands that has highest number of incidents (Bar graph) ##

# Renamed column to expand abbreviation
df = df.rename(columns={'PVE_NAAM': 'Province Name'})

# Count the incidents by province
incident_by_province = df['Province Name'].value_counts()
# Plotting incidents by province
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Province Name', order=incident_by_province.index)
plt.title('Cycling Incidents by Province in the Netherlands')
plt.xticks(rotation=90)
plt.ylabel('Number of Incidents')

# Create output folder to save visual
viz_output_folder = Path(
    r"C:\Users\ThinkBook\Desktop\netherlands\six4change-quant-data\visualisations")
viz_output_folder.mkdir(parents=True, exist_ok=True)

# Save bar graph
viz_path = viz_output_folder / "Cycling_Incidents_by_Netherlands_province.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---------------------------------------------------------------------------#

## How the highest number of incidents happening in Amsterdam can be caused by different built-up areas (pie chart) ##

# Print the names of the neighborhoods in the Netherlands
print(df['WIK_NAAM'].unique())

# List of Amsterdam neighborhoods
amsterdam_neighborhoods = [
    'AMSTERDAM-CENTRUM', 'OUD-WEST', 'OUD ZUID', 'GEUZENVELD/SLOTERMEER',
    'OSDORP', 'SLOTERVAART/OVERTOOMSE VELD', 'DE BAARSJES', 'ZUIDOOST',
    'BOS EN LOMMER', 'WESTPOORT/SLOTERDIJK'
]

# Filter the data for only Amsterdam neighborhoods
amsterdam_df = df[df['WIK_NAAM'].isin(amsterdam_neighborhoods)]

# Save cleaned data for Amsterdam to dataset folder
csv_output_folder = Path(
    r"C:\Users\ThinkBook\Desktop\netherlands\csv tables")
csv_output_folder.mkdir(parents=True, exist_ok=True)
cleaned_csv_path = csv_output_folder / "cleaned_cycling_safety_amsterdam.csv"
df.to_csv(cleaned_csv_path, index=False)

# Plotting a Built-up Area (BEBKOM) Pie Chart (Inside vs Outside)
plt.figure(figsize=(8, 8))
become_counts = amsterdam_df['BEBKOM'].value_counts()
become_labels = ['Inside Built-up Area (BI)', 'Outside Built-up Area (BU)']
plt.pie(become_counts, labels=become_labels, autopct='%1.1f%%',
        startangle=90, colors=['#66b3ff', '#99ff99'])
plt.title('Distribution of Incidents by Built-up Area in Amsterdam')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Create output folder to save pie chart
viz_output_folder = Path(
    r"C:\Users\ThinkBook\Desktop\netherlands\six4change-quant-data\visualisations")
viz_output_folder.mkdir(parents=True, exist_ok=True)

# Save pie chart
viz_path = viz_output_folder / "Most_common_Built-up_Area_type_in_Amsterdam.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---------------------------------------------------------------------------#

## How does different Pavement Types (WVG_AN) influence the number of incidents (horizontal bar graph) ##

# Change origin to uppercase and strip spaces
amsterdam_df['WVG_AN_norm'] = amsterdam_df['WVG_AN'].str.upper().str.strip()

# Print all types of road pavements
print(df['WVG_AN'].unique())

# Categorize the road pavement types
pavement_categories_expanded = {
    # Dedicated bike paths
    'FIETSPAD': 'Dedicated Bike Paths',
    'FIETSPAD ASFALT': 'Dedicated Bike Paths',
    'NORMAAL FIETSPAD': 'Dedicated Bike Paths',
    'TEGLFIETSPAD': 'Dedicated Bike Paths',

    # Red tile bike paths
    'RODE FIETSPAD TEGELS': 'Red Tile Bike Paths',

    # Cobblestone & Asphalt
    'KLINKERS': 'Cobblestone & Asphalt Mix',
    'KLINKERS EN ASFALT': 'Cobblestone & Asphalt Mix',
    'ASFALT': 'Cobblestone & Asphalt Mix',
    'KEIEN': 'Cobblestone & Asphalt Mix',
    'STENEN': 'Cobblestone & Asphalt Mix',
    'STRAATSTEEN': 'Cobblestone & Asphalt Mix',
    'STRAATSTEENEN': 'Cobblestone & Asphalt Mix',
    'STRAATTEGEL 30/30': 'Sidewalk Tiles & Pavement',

    # Loose surfaces & gravel
    'GRAVEL': 'Loose Surfaces & Gravel',
    'GRIND': 'Loose Surfaces & Gravel',
    'GRINDPAD': 'Loose Surfaces & Gravel',
    'GRINT': 'Loose Surfaces & Gravel',
    'GRIND SCHELPEN': 'Loose Surfaces & Gravel',
    'SCHELPEN': 'Loose Surfaces & Gravel',
    'SCHELPENPAD': 'Loose Surfaces & Gravel',
    'GEPERST GRAVEL': 'Loose Surfaces & Gravel',
    'FIETSPASGRAVEL': 'Loose Surfaces & Gravel',

    # Sidewalk tiles & pavement
    'TEGELS': 'Sidewalk Tiles & Pavement',
    'STOEPTEGELS': 'Sidewalk Tiles & Pavement',
    'TROTTOIRTEGELS': 'Sidewalk Tiles & Pavement',
    'STRAATTEGELS': 'Sidewalk Tiles & Pavement',
    'STRAATTEGEL': 'Sidewalk Tiles & Pavement',
    'TROTTOIR TEGELS': 'Sidewalk Tiles & Pavement',
    'STOEPTEGELS': 'Sidewalk Tiles & Pavement',
    'TROTTOIRTEGELS': 'Sidewalk Tiles & Pavement',
    'STRAATTEGELS': 'Sidewalk Tiles & Pavement',
    'TROTTOIR TEGELS': 'Sidewalk Tiles & Pavement',

    # Unpaved, sand, wood
    'ZAND': 'Unpaved, Sand & Wood',
    'ZANDPAD': 'Unpaved, Sand & Wood',
    'ZANDWEG': 'Unpaved, Sand & Wood',
    'ONVERHARDE WEG': 'Unpaved, Sand & Wood',
    'ONVERHARD': 'Unpaved, Sand & Wood',
    'BOSPAD': 'Unpaved, Sand & Wood',
    'BOSPAD ZAND': 'Unpaved, Sand & Wood',
    'HOUTENBRUG MET ANTISLIP LAAG': 'Unpaved, Sand & Wood',
    'RIJPLATEN': 'Unpaved, Sand & Wood',
    'GRAS': 'Unpaved, Sand & Wood',

    # Train/Tram tracks
    'RAILS': 'Train/Tram Tracks',
    'SPOOR': 'Train/Tram Tracks',
    'SPOORWEGOVERGANG': 'Train/Tram Tracks',
}

# Mapping normalized pavement types
amsterdam_df['Pavement_Category'] = amsterdam_df['WVG_AN_norm'].map(
    pavement_categories_expanded)

# Filter out rows with unmapped pavement types (with NaN values)
amsterdam_df_filtered = amsterdam_df.dropna(
    subset=['Pavement_Category']).copy()

# Group and aggregate
pavement_counts = amsterdam_df_filtered.groupby(
    'Pavement_Category').size().reset_index(name='Count')
pavement_counts = pavement_counts.sort_values(by='Count', ascending=False)

# Plotting the pavement types vs number of incidents
plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Pavement_Category',
            data=pavement_counts, palette='Set2')
plt.title('Cycling Incidents by Road Pavement Type in Amsterdam')
plt.xlabel('Number of Incidents')
plt.ylabel('Type of Road Pavement')
plt.tight_layout()

# Create output folder to save visual
viz_output_folder = Path(
    r"C:\Users\ThinkBook\Desktop\netherlands\six4change-quant-data\visualisations")
viz_output_folder.mkdir(parents=True, exist_ok=True)

# Save horizontal bar graph
viz_path = viz_output_folder / \
    "Cycling_Incidents_by_road_pavement_types_in_Amsterdam.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---------------------------------------------------------------------------#

## How different Pavement Types (WVG_AN) impact the severity of the incident (horizontal bar graph) ##

# Grouping by Pavement Category and calculating metrics
severity_analysis = amsterdam_df_filtered.groupby('Pavement_Category').agg(
    Total_Incidents=('Injuries', 'size'),
    Avg_Severity=('Injuries', 'mean')
).reset_index()

# Plotting Average Severity per Pavement Type
plt.figure(figsize=(12, 6))
sns.barplot(data=severity_analysis, x='Avg_Severity',
            y='Pavement_Category', palette='coolwarm')
plt.title('Average Severity by Road Pavement Type')
plt.xlabel('Level of Severity of Injuries')
plt.ylabel('Type of Road Pavement')
plt.tight_layout()

# Save horizontal bar graph
viz_path = viz_output_folder / \
    "Average_Severity_by_Pavement_Category_in_Amsterdam.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---------------------------------------------------------------------------#

## How built-up area change overtime influence the number of incidents (line graph) ##

# For builtup category trend
builtup_yearly = amsterdam_df.groupby(
    ['JAAR_VKL', 'BEBKOM']).size().reset_index(name='Incidents')

# For pavement category trend
pavement_yearly = amsterdam_df_filtered.groupby(
    ['JAAR_VKL', 'Pavement_Category']).size().reset_index(name='Incidents')

# Plot for Built-up Area
plt.figure(figsize=(12, 6))
sns.lineplot(data=builtup_yearly, x='JAAR_VKL',
             y='Incidents', hue='BEBKOM', marker='o')
plt.title('Cycling Incidents over Years by Built-up Area in Amsterdam')
plt.ylabel('Number of Incidents')
plt.xlabel('Year')
plt.legend(title='Built-up Area')
plt.grid(True)
plt.tight_layout()

# Create output folder to save visual
viz_output_folder = Path(
    r"C:\Users\ThinkBook\Desktop\netherlands\six4change-quant-data\visualisations")
viz_output_folder.mkdir(parents=True, exist_ok=True)

# Save line graph
viz_path = viz_output_folder / \
    "Cycling_Incidents_over_Years_by_Built-up_Area_in_Amsterdam.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---------------------------------------------------------------------------#

## How different Pavement Types (WVG_AN) change overtime influence the number of incidents (stacked bar + line graph) ##

# Pivot for stacked bar chart
pivot_df = pavement_yearly.pivot(
    index='JAAR_VKL', columns='Pavement_Category', values='Incidents').fillna(0)

# Use seaborn color palette
categories = pivot_df.columns.tolist()
palette = sns.color_palette('Set2', n_colors=len(categories))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

# 1) Stacked bar chart on ax1
bottom = pd.Series([0]*len(pivot_df), index=pivot_df.index)
for i, category in enumerate(categories):
    ax1.bar(pivot_df.index, pivot_df[category],
            bottom=bottom, color=palette[i], label=category)
    bottom += pivot_df[category]

ax1.set_title(
    'Cycling Incidents by Road Pavement Type Over Years')
ax1.set_ylabel('Number of Incidents')
ax1.legend(title='Type of Road Pavement',
           bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True)

# 2) Line chart on ax2 for trends
for i, category in enumerate(categories):
    ax2.plot(pivot_df.index, pivot_df[category],
             marker='o', color=palette[i], label=category)

ax2.set_title(
    'Trend of Cycling Incidents by Road Pavement Type Over Years')
ax2.set_ylabel('Number of Incidents')
ax2.set_xlabel('Year')
ax2.grid(True)

# Remove duplicate legends from ax2
ax2.legend().set_visible(False)

plt.tight_layout()

# Save stacked bar + line graph
viz_output_folder = Path(
    r"C:\Users\ThinkBook\Desktop\netherlands\six4change-quant-data\visualisations")
viz_output_folder.mkdir(parents=True, exist_ok=True)
viz_path = viz_output_folder / \
    "Cycling_Incidents_over_Years_by_Pavement_Category_in_Amsterdam.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
