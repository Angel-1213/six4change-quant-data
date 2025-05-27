from scipy.stats import chi2_contingency
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# File paths
ongevallen_txt_2023 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\data analysis\2023_Ongevallengegevens\2023_ongevallen.txt'
partijen_txt_2023 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\data analysis\2023_Ongevallengegevens\2023_partijen.txt'
juncties_path_2023 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\data analysis\2023_Netwerkgegevens\2023_juncties.txt'
wegvakken_path_2023 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\data analysis\2023_Netwerkgegevens\2023_wegvakken.txt'

# Data source: https://nationaalgeoregister.nl/geonetwork/srv/dut/catalog.search#/metadata/4gqrs90k-vobr-5t59-x726-4x2unrs1vawz

# --- Data analysis for 2023 ---#

# Step 1: Process accident data
chunksize = 100_000
chunks = []

for chunk in pd.read_csv(ongevallen_txt_2023, chunksize=chunksize, low_memory=False):
    filtered_chunk = chunk[['VKL_NUMMER',
                            'PVE_NAAM', 'JTE_ID', 'JAAR_VKL']].copy()
    filtered_chunk.rename(columns={
        'VKL_NUMMER': 'Accident_ID',
        'PVE_NAAM': 'Province',
        'JTE_ID': 'Intersection_ID',
        'JAAR_VKL': 'Year'
    }, inplace=True)
    chunks.append(filtered_chunk)

df_provinces = pd.concat(chunks, ignore_index=True)

# Step 2: Load parties data
df_parties = pd.read_csv(partijen_txt_2023, low_memory=False)
cyclist_types = [
    'FATBIKE', 'FAT BIKE',
    'SPEEDPEDELEC', 'SPEED PEDELEC'
]
df_cyclists = df_parties[df_parties['OTE_AN'].isin(cyclist_types)][['VKL_NUMMER', 'OTE_AN']].rename(columns={
    'VKL_NUMMER': 'Accident_ID',
    'OTE_AN': 'Party_Type'
})

# Step 3: Merge cyclist and province data
df_cyclist_accidents = pd.merge(
    df_cyclists, df_provinces, on='Accident_ID', how='left')

# Step 4: Add juncties info
df_juncties = pd.read_csv(juncties_path_2023, low_memory=False)
df_juncties_renamed = df_juncties[['JTE_ID', 'WBRSRT_R', 'WBRSRT_G', 'WBRSRT_W', 'SLE_TYPE', 'SLE_NUMMER']].rename(columns={
    'WBRSRT_R': 'Main_Road_Type',
    'WBRSRT_G': 'Urban_Access_Road_Type',
    'WBRSRT_W': 'Residential_Road_Type',
    'SLE_TYPE': 'Intersection_Type',
    'SLE_NUMMER': 'Road_Segment_Number'
})

df_final = pd.merge(df_cyclist_accidents, df_juncties_renamed,
                    left_on='Intersection_ID', right_on='JTE_ID', how='left')
df_final.drop(columns=['JTE_ID'], inplace=True)

# Step 5: Add wegvakken info
wegvakken_cols = ['BST_CODE', 'RPE_CODE', 'WEGBEHSRT', 'SLE_NUMMER']
wegvakken_renamed_cols = {
    'BST_CODE': 'Pavement_Type',
    'RPE_CODE': 'Surface_Type',
    'WEGBEHSRT': 'Road_Structure_Type',
    'SLE_NUMMER': 'Road_Segment_Number'
}

chunks = []
for chunk in pd.read_csv(wegvakken_path_2023, usecols=wegvakken_cols, chunksize=100_000, low_memory=False):
    chunk.rename(columns=wegvakken_renamed_cols, inplace=True)
    chunks.append(chunk)

df_wegvakken = pd.concat(chunks, ignore_index=True).drop_duplicates(
    subset=['Road_Segment_Number'])
df_final = pd.merge(df_final, df_wegvakken,
                    on='Road_Segment_Number', how='left')

# Save merged data to csv tables
output_path = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\csv tables\2023_merged_data_Netherlands.csv'
df_final.to_csv(output_path, index=False)

# Step 6: Filter for Noord-Holland and Save

# Load merged dataset
merged_output_path = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\csv tables\2023_merged_data_Netherlands.csv'
df = pd.read_csv(merged_output_path)


# Filter for Noord-Holland
df_nh = df[df['Province'] == 'Noord-Holland'].copy()

# Define path for filtered output
noord_holland_output_path = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\csv tables\2023_merged_data_Amsterdam.csv'

# Save filtered data
df_nh.to_csv(noord_holland_output_path, index=False)

# Step 7: Filtering wegvakken data （pavement and surface types) and plot with number of cycling incidents

# Print unique pavement types
print("Unique pavement types:", df_nh['Pavement_Type'].unique())

# Count number of incidents per pavement type (count each accident only once)
pavement_counts = (
    df_nh.drop_duplicates('Accident_ID')
         .groupby('Pavement_Type')
         .size()
         .reset_index(name='Accident_Count')
         .sort_values('Accident_Count', ascending=False)
)

pavement_labels = {
    'PKP': 'Parking Access Road',
    'FP': 'Bicycle Path',
    'RB': 'Carriageway',
    'VBW': 'Connecting Road',
    'NRB': 'Roundabout Lane'
}

# Map codes to full names
pavement_counts['Pavement_Type_Full'] = pavement_counts['Pavement_Type'].map(
    pavement_labels)

# Plotting bar graph with full names
plt.figure(figsize=(10, 6))
sns.barplot(data=pavement_counts, x='Pavement_Type_Full',
            y='Accident_Count', palette='pastel')

# Add labels and title
plt.xlabel('Road Pavement Type')
plt.ylabel('Number of Cycling Incidents')
plt.title('Cycling Incidents by Road Pavement Type in Amsterdam ')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save bar graph to visualisations folder
plt.savefig(r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\visualisations\2023_cycling_incidents_by_pavement_type_Amsterdam.png')

# Print unique road surface types
print("Unique surface types:", df_nh['Surface_Type'].unique())

# Count number of incidents per road surface type (count each accident only once)
surface_counts = (
    df_nh.drop_duplicates('Accident_ID')
         .groupby('Surface_Type')
         .size()
         .reset_index(name='Accident_Count')
         .sort_values('Accident_Count', ascending=False)
)

surface_labels = {
    'O': 'Unpaved',
    'N': 'Paved',
}

# Map codes to full names
surface_counts['Surface_Type_Full'] = surface_counts['Surface_Type'].map(
    surface_labels)

# Plotting bar graph
plt.figure(figsize=(10, 6))
sns.barplot(data=surface_counts, x='Surface_Type_Full',
            y='Accident_Count', palette='pastel')

# Add labels and title
plt.xlabel('Road Surface Type')
plt.ylabel('Number of Cycling Incidents')
plt.title('Cycling Incidents by Road Surface Type in Amsterdam ')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save bar graph to visualisations folder
plt.savefig(r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\visualisations\2023_cycling_incidents_by_surface_type_Amsterdam.png')

# Step 7: Create contingency table: Pavement_Type vs. Incident_Count
contingency_table_p = pd.crosstab(df_nh['Pavement_Type'], columns='count')

# Perform chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table_p)
print(f"p-value: {p:.4f}")

# Create contingency table: Surface_Type vs. Incident_Count
contingency_table_s = pd.crosstab(df_nh['Surface_Type'], columns='count')

# Perform chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table_s)
print(f"p-value: {p:.4f}")
# ______________________________________________________________________________________________________________________________________ #

# --- File paths for 2021 ---
ongevallen_txt_2021 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\data analysis\2021_Ongevallengegevens\2021_ongevallen.txt'
partijen_txt_2021 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\data analysis\2021_Ongevallengegevens\2021_partijen.txt'
juncties_path_2021 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\data analysis\2021_Netwerkgegevens\2021_juncties.txt'
wegvakken_path_2021 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\data analysis\2021_Netwerkgegevens\2021_wegvakken.txt'

output_path_2021 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\csv tables\2021_merged_data_Netherlands.csv'
filtered_output_path_2021 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\csv tables\2021_merged_data_Amsterdam.csv'

# --- Step 1: Read ongevallen data ---
df_ongevallen_2021 = pd.read_csv(
    ongevallen_txt_2021,
    encoding='latin1',
    sep=',',
    quotechar='"',
    low_memory=False,
    usecols=['VKL_NUMMER', 'PVE_NAAM', 'JTE_ID', 'JAAR_VKL'],
    dtype={'VKL_NUMMER': str, 'PVE_NAAM': str, 'JTE_ID': str, 'JAAR_VKL': int}
).rename(columns={
    'VKL_NUMMER': 'Accident_ID',
    'PVE_NAAM': 'Province',
    'JTE_ID': 'Intersection_ID',
    'JAAR_VKL': 'Year'
})

# Sanity check: Confirm year values are 2021
assert set(df_ongevallen_2021['Year'].unique()) == {
    2021}, "Year column contains unexpected years"

# --- Step 2: Read partijen data and filter cyclists ---
df_partijen_2021 = pd.read_csv(partijen_txt_2021, low_memory=False, dtype={
                               'VKL_NUMMER': str, 'OTE_AN': str})
cyclist_types = ['FATBIKE', 'FAT BIKE', 'SPEEDPEDELEC', 'SPEED PEDELEC']
df_cyclists_2021 = df_partijen_2021[df_partijen_2021['OTE_AN'].isin(cyclist_types)][['VKL_NUMMER', 'OTE_AN']].rename(columns={
    'VKL_NUMMER': 'Accident_ID',
    'OTE_AN': 'Party_Type'
})


# --- Step 3: Merge cyclist data with ongevallen (province and intersection info) ---
df_cyclist_accidents_2021 = pd.merge(
    df_cyclists_2021,
    df_ongevallen_2021[['Accident_ID', 'Province', 'Intersection_ID', 'Year']],
    on='Accident_ID',
    how='left'
)

# --- Step 4: Read juncties data and rename columns ---
df_juncties_2021 = pd.read_csv(juncties_path_2021, low_memory=False, dtype=str)
df_juncties_renamed_2021 = df_juncties_2021[['JTE_ID', 'WBRSRT_R', 'WBRSRT_G', 'WBRSRT_W', 'SLE_TYPE', 'SLE_NUMMER']].rename(columns={
    'WBRSRT_R': 'Main_Road_Type',
    'WBRSRT_G': 'Urban_Access_Road_Type',
    'WBRSRT_W': 'Residential_Road_Type',
    'SLE_TYPE': 'Intersection_Type',
    'SLE_NUMMER': 'Road_Segment_Number'
})

# Merge juncties data into cyclist accidents
df_final_2021 = pd.merge(
    df_cyclist_accidents_2021,
    df_juncties_renamed_2021,
    left_on='Intersection_ID',
    right_on='JTE_ID',
    how='left'
)
df_final_2021.drop(columns=['JTE_ID'], inplace=True)

# --- Step 5: Read wegvakken data in chunks and rename columns ---
wegvakken_cols = ['BST_CODE', 'RPE_CODE', 'WEGBEHSRT', 'SLE_NUMMER']
wegvakken_renamed_cols = {
    'BST_CODE': 'Pavement_Type',
    'RPE_CODE': 'Surface_Type',
    'WEGBEHSRT': 'Road_Structure_Type',
    'SLE_NUMMER': 'Road_Segment_Number'
}

wegvakken_chunks = []

with open(wegvakken_path_2021, 'r', encoding='latin1', errors='replace') as f:
    for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False, sep=',', quotechar='"'):
        chunk = chunk[wegvakken_cols].rename(columns=wegvakken_renamed_cols)
        wegvakken_chunks.append(chunk)

df_wegvakken_2021 = pd.concat(wegvakken_chunks, ignore_index=True).drop_duplicates(
    subset=['Road_Segment_Number'])

# Convert both columns to string type before merge to avoid dtype mismatch
df_final_2021['Road_Segment_Number'] = df_final_2021['Road_Segment_Number'].astype(
    str)
df_wegvakken_2021['Road_Segment_Number'] = df_wegvakken_2021['Road_Segment_Number'].astype(
    str)

# Merge wegvakken into final dataframe
df_final_2021 = pd.merge(
    df_final_2021,
    df_wegvakken_2021,
    on='Road_Segment_Number',
    how='left'
)

# --- Step 6: Save the full merged data for 2021 ---
df_final_2021.to_csv(output_path_2021, index=False)

# --- Step 7: Filter for 'Noord-Holland' province and save ---
df_nh_2021 = df_final_2021[df_final_2021['Province'] == 'Noord-Holland'].copy()
df_nh_2021.to_csv(filtered_output_path_2021, index=False)

# ______________________________________________________________________________________________________________________________________ #

# --- Cleaning datasets from 2022 ---#


# File paths for 2022 data
ongevallen_txt_2022 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\data analysis\2022_Ongevallengegevens\ongevallen.txt'
partijen_txt_2022 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\data analysis\2022_Ongevallengegevens\partijen.txt'
juncties_path_2022 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\data analysis\2022_Netwerkgegevens\juncties.txt'
wegvakken_path_2022 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\data analysis\2022_Netwerkgegevens\wegvakken.txt'

# Step 1: Read accidents data
df_ongevallen_2022 = pd.read_csv(
    ongevallen_txt_2022,
    encoding='latin1',
    sep=',',
    quotechar='"',
    low_memory=False
)

# Step 2: Read parties data and filter cyclists
df_parties_2022 = pd.read_csv(partijen_txt_2022, low_memory=False)
cyclist_types = ['FATBIKE', 'FAT BIKE', 'SPEEDPEDELEC', 'SPEED PEDELEC']
df_cyclists_2022 = df_parties_2022[df_parties_2022['OTE_AN'].isin(cyclist_types)][['VKL_NUMMER', 'OTE_AN']].rename(columns={
    'VKL_NUMMER': 'Accident_ID',
    'OTE_AN': 'Party_Type'
})

# Step 3: Prepare provinces dataframe from accidents (rename relevant columns)
df_provinces_2022 = df_ongevallen_2022[['VKL_NUMMER', 'PVE_NAAM', 'JTE_ID', 'JAAR_VKL']].rename(columns={
    'VKL_NUMMER': 'Accident_ID',
    'PVE_NAAM': 'Province',
    'JTE_ID': 'Intersection_ID',
    'JAAR_VKL': 'Year'
})

# Merge cyclists and provinces on Accident_ID
df_cyclist_accidents_2022 = pd.merge(
    df_cyclists_2022,
    df_provinces_2022,
    on='Accident_ID',
    how='left'
)

# Step 4: Add juncties info
df_juncties_2022 = pd.read_csv(juncties_path_2022, low_memory=False)
df_juncties_renamed_2022 = df_juncties_2022[['JTE_ID', 'WBRSRT_R', 'WBRSRT_G', 'WBRSRT_W', 'SLE_TYPE', 'SLE_NUMMER']].rename(columns={
    'WBRSRT_R': 'Main_Road_Type',
    'WBRSRT_G': 'Urban_Access_Road_Type',
    'WBRSRT_W': 'Residential_Road_Type',
    'SLE_TYPE': 'Intersection_Type',
    'SLE_NUMMER': 'Road_Segment_Number'
})

df_final_2022 = pd.merge(df_cyclist_accidents_2022, df_juncties_renamed_2022,
                         left_on='Intersection_ID', right_on='JTE_ID', how='left')
df_final_2022.drop(columns=['JTE_ID'], inplace=True)

# Step 5: Add wegvakken info with chunks
chunksize = 100000  # adjust if needed
wegvakken_cols = ['BST_CODE', 'RPE_CODE', 'WEGBEHSRT', 'SLE_NUMMER']
wegvakken_renamed_cols = {
    'BST_CODE': 'Pavement_Type',
    'RPE_CODE': 'Surface_Type',
    'WEGBEHSRT': 'Road_Structure_Type',
    'SLE_NUMMER': 'Road_Segment_Number'
}

wegvakken_chunks = []
with open(wegvakken_path_2022, 'r', encoding='latin1', errors='replace') as f:
    for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False, sep=',', quotechar='"'):
        chunk = chunk[wegvakken_cols].rename(columns=wegvakken_renamed_cols)
        wegvakken_chunks.append(chunk)

df_wegvakken_2022 = pd.concat(wegvakken_chunks, ignore_index=True).drop_duplicates(
    subset=['Road_Segment_Number'])

# Fix dtype mismatch before merging
df_final_2022['Road_Segment_Number'] = df_final_2022['Road_Segment_Number'].astype(
    str)
df_wegvakken_2022['Road_Segment_Number'] = df_wegvakken_2022['Road_Segment_Number'].astype(
    str)

df_final_2022 = pd.merge(df_final_2022, df_wegvakken_2022,
                         on='Road_Segment_Number', how='left')

# Step 6: Save merged data to CSV
output_path_2022 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\csv tables\2022_merged_data_Netherlands.csv'
df_final_2022.to_csv(output_path_2022, index=False)

# Step 7: Filter for Noord-Holland and save filtered dataset
df_nh_2022 = df_final_2022[df_final_2022['Province'] == 'Noord-Holland'].copy()
noord_holland_output_path_2022 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\csv tables\2022_merged_data_Amsterdam.csv'
df_nh_2022.to_csv(noord_holland_output_path_2022, index=False)


# ______________________________________________________________________________________________________________________________________ #

# --- Data analysis from 2021-2023 ---#

# File paths
file_2021 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\csv tables\2021_merged_data_Amsterdam.csv'
file_2022 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\csv tables\2022_merged_data_Amsterdam.csv'
file_2023 = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\csv tables\2023_merged_data_Amsterdam.csv'

# Step 1: Read CSVs
df_2021 = pd.read_csv(file_2021)
df_2022 = pd.read_csv(file_2022)
df_2023 = pd.read_csv(file_2023)

# Step 2: Merge all three csv and save
# Concatenate all rows from the three DataFrames
df_from_2021_to_2023 = pd.concat(
    [df_2021, df_2022, df_2023], ignore_index=True)

# Output file path
output_path = r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\csv tables\2021_to_2023_merged_data_Amsterdam.csv'

# Save merged data
df_from_2021_to_2023.to_csv(output_path, index=False)

# Step 3: Plotting line graph with road and surface pavement against number of cycling incidents

# Group by Year and Pavement Type
pavement_year_counts = (
    df_from_2021_to_2023.groupby(['Year', 'Pavement_Type'])
    .size()
    .reset_index(name='Accident_Count')
)

# Map pavement codes to readable labels
pavement_labels = {
    'PKP': 'Parking Access Road',
    'FP': 'Bicycle Path',
    'RB': 'Carriageway',
    'VBW': 'Connecting Road',
    'NRB': 'Roundabout Lane'
}
pavement_year_counts['Pavement_Type_Full'] = pavement_year_counts['Pavement_Type'].map(
    pavement_labels
)

# Convert Year to numeric type
pavement_year_counts['Year'] = pd.to_numeric(
    pavement_year_counts['Year'], errors='coerce')

# Drop rows where Year conversion failed (just in case)
pavement_year_counts = pavement_year_counts.dropna(subset=['Year'])
pavement_year_counts['Year'] = pavement_year_counts['Year'].astype(int)

# Create a complete grid of Year and Pavement_Type
all_years = [2021, 2022, 2023]
all_pavements = pavement_year_counts['Pavement_Type_Full'].unique()
complete_grid = pd.DataFrame([
    (year, pavement)
    for year in all_years
    for pavement in all_pavements
], columns=['Year', 'Pavement_Type_Full'])

# Merge with actual data and fill missing counts with 0
pavement_year_counts_complete = complete_grid.merge(
    pavement_year_counts,
    on=['Year', 'Pavement_Type_Full'],
    how='left'
).fillna({'Accident_Count': 0})

# Ensure 'Year' is treated as integer
pavement_year_counts_complete['Year'] = pavement_year_counts_complete['Year'].astype(
    int)

# Plot as grouped bar chart
plt.figure(figsize=(12, 6))
sns.barplot(
    data=pavement_year_counts_complete,
    x='Year',
    y='Accident_Count',
    hue='Pavement_Type_Full',
    palette='pastel'
)

# Labels and formatting
plt.xlabel('Year')
plt.ylabel('Number of Cycling Incidents')
plt.title('Cycling Incidents by Pavement Type (2021–2023)')
plt.legend(title='Road Type')
plt.xticks([0, 1, 2], [2021, 2022, 2023])
plt.tight_layout()

# Save bar graph to visualisations folder
plt.savefig(r'C:\Users\ThinkBook\Desktop\six4change-quant-data-main\six4change-quant-data-main\visualisations\2021_to_2023_cycling_incidents_by_pavement_type_Amsterdam.png')
