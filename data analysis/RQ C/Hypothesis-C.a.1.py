from pathlib import Path
import pandas as pd
import statsmodels.api as sm

print("The Research question being answered is: What travel motive do cyclists in the Netherlands cycle the longest duration for, and how does age influence this?")
print("The hypothesis being tested is: There is a significant difference in average cycling distance between cycling for leisure and cycling to commute to work.")

# 1. CONFIGURATION
# Define all paths upfront
DATA_PATH = Path(
    r"..\csv tables\transport_per_motive.csv")
OUTPUT_FOLDER_CSV = Path(
    r"..\csv tables")
OUTPUT_FOLDER_VIS = Path(
    r"..\visualisations\C.a.1"
    "")

# Ensure output folders exist
for folder in [OUTPUT_FOLDER_CSV, OUTPUT_FOLDER_VIS]:
    folder.mkdir(parents=True, exist_ok=True)


# 2. DATA PROCESSING

def load_and_clean_data(file_path):
    """Load and clean the raw data"""
    data = pd.read_csv(file_path)

    # Rename columns to english
    data.rename(columns={ 
    'Geslacht': 'Gender',
    'Leeftijd': 'Age',
    'Vervoerwijzen': 'Mode of transport',
    'Reismotieven': 'Travel motives',
    'Marges': 'Margins',
    'Perioden': 'Year',
    'Ritten per persoon per dag (gemiddeld) (aantal)': 'Average trips per person per day',
    'Afgelegde afstand per rit (gemiddeld) (km)': 'Average travel distance per trip (km)',
    'Reisduur per rit (Minuten)': 'Average traveltime per tip (minutes)',
    }, inplace=True)

    # Remove unnecessary columns
    data.drop(columns=['Mode of transport'], inplace=True)
    data.drop(columns=['Year'], inplace=True)
    data.drop(columns=['Gender'], inplace=True)
    
    # Rename column values to english
    
    # Translations for travel motives
    data['Travel motives'].replace({
    'Totaal': 'Total',
    'Onderwijs volgen, cursus, kinderopvang': 'Education, courses, childcare',
    'Overige reismotieven': 'Other travel motives',
    'Van en naar het werk': 'Commute to and from work',
    'Vrije tijd': 'Leisure',
    'Winkelen, boodschappen doen': 'Shopping, doing groceries'
    }   , inplace=True)

    # Translations for age groups
    data['Age'].replace({
    'Totaal': 'Total',
    '6 tot 12 jaar': '6-12 years',
    '12 tot 18 jaar': '12-18 years',
    '18 tot 25 jaar': '18-25 years',
    '25 tot 30 jaar': '25-30 years',
    '30 tot 40 jaar': '30-40 years',
    } , inplace=True)

    # Filter only 'Value' rows
    bicycle_data = data[data['Margins'] == 'Waarde'].copy()

    # Remove 'Total' rows from 'Travel motives' and 'Age' columns
    bicycle_data = bicycle_data[bicycle_data['Travel motives'] != 'Total']
    bicycle_data = bicycle_data[bicycle_data['Age'] != 'Total']

    # Drop rows with any NaN values
    bicycle_data.dropna(inplace=True)

    # Reset index after filtering and dropping NaNs
    bicycle_data.reset_index(drop=True, inplace=True)

    # Convert European number formats
    numeric_cols = [
       'Average trips per person per day',
       'Average travel distance per trip (km)',
        'Average traveltime per tip (minutes)',
    ]

    for col in numeric_cols:
        bicycle_data[col] = (
            bicycle_data[col]
            .astype(str)
            .str.replace(',', '.')
            .astype(float)
        )
    

    # Calculate average travel distance per day
    bicycle_data['Average travel distance (km) per day'] = (
    bicycle_data['Average travel distance per trip (km)'] *
    bicycle_data['Average trips per person per day']
    ) 


    return bicycle_data


# Filter and binary encode Travel motives
def binary_encode_travel_motives(data):
    # Filter to include only the two travel motives of interest
    filtered = data[data['Travel motives'].isin(['Commute to and from work', 'Leisure'])].copy()

    # bbinary encode Work = 0, Leisure = 1
    motive_map = {
        'Commute to and from work': 0,
        'Leisure': 1
    }
    # Apply the mapping to create a new binary column
    filtered['Travel motive (binary)'] = filtered['Travel motives'].map(motive_map)

    return filtered


# 3. LINEAR REGRESSION
def linear_regression(data, y, X):
    # Define predictors (all one-hot columns starting with 'Travel motives_')
    X = sm.add_constant(data[X]) # Add the binary column for intercept

    # Define response variable
    y = data[y]

    # Add constant to the model (for intercept)
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X).fit()

    with open(OUTPUT_FOLDER_VIS / 'regression_model_summary.html', 'w') as f:
        f.write(model.summary().as_html())

    return model

#EXECUTION
# 2. Load and clean data
bicycle_data = load_and_clean_data(DATA_PATH)

# Binary encode Travel Motives
motive_binary = binary_encode_travel_motives(bicycle_data)

# 3. Run linear regression
model = linear_regression(motive_binary, 'Average travel distance (km) per day', 'Travel motive (binary)')

# Save processed data and model summary
motive_binary.to_csv(OUTPUT_FOLDER_CSV / 'processed_travel_data.csv', index=False)