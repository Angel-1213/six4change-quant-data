from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

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


# add a row with the avrages of each column
def average(data):
    # Columns to average
    cols_to_average = [
    'Average trips per person per day',
    'Average travel distance per trip (km)',
    'Average traveltime per tip (minutes)',
    ]

    # Create a dictionary with the column-wise averages
    average_row = data[cols_to_average].mean().to_dict()

    # Optionally fill in identifiers so the row makes sense
    average_row.update({
    'Travel motives': 'Average of motives',
    'Age': 'Average of ages',
    'Margins': 'Waarde'
    })

    # Append to the DataFrame
    data = pd.concat([data, pd.DataFrame([average_row])], ignore_index=True)  
    return data

# One hot encoding for categorical variables
def one_hot_encode(data, column):
   encoder = OneHotEncoder(sparse_output=False, drop=['Shopping, doing groceries']) # Initialise the encoder

   one_hot_encoded = encoder.fit_transform(data[[column]]) # Fit the encder to the data and transform it

   one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(), index=data.index) # Create a DataFrame from the encoded data

   df_encoded = pd.concat([data, one_hot_df], axis=1).dropna() # Concatenate the original DataFrame with the one-hot encoded DataFrame

   df_encoded = df_encoded.drop(columns=[column]) # Drop the original column after encoding
   return df_encoded # Return the DataFrame with the one-hot encoded columns

# 3. LINEAR REGRESSION
def linear_regression(data, y, X):
    # Define predictors (all one-hot columns starting with 'Travel motives_')
    X = data.filter(like='Travel motives_')

    # Define response variable
    y = data['Average travel distance (km) per day']

    # Add constant to the model (for intercept)
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Define the contrast: Leisure coefficient - Commute coefficient = 0
    contrast = 'Travel motives_Leisure - Travel motives_Commute to and from work = 0'

    # Perform t-test
    t_test_result = model.t_test(contrast)

    return model, t_test_result

# 4. VISUALISATIONS

def visualisations(data, model):
    # Create a summary table of the regression results
    summary_table = model.summary()
    summary_table_as_html = summary_table.as_html()

    # Save the summary table as an HTML file
    with open(OUTPUT_FOLDER_VIS / 'regression_summary.html', 'w') as f:
        f.write(summary_table_as_html)

    # Create a bar plot of the coefficients
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.params.index[1:], y=model.params.values[1:])
    plt.title('Regression Coefficients')
    plt.xlabel('Travel Motives')
    plt.ylabel('Coefficient Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER_VIS / 'regression_coefficients.png')

#EXECUTION
bicycle_data = load_and_clean_data(DATA_PATH)

averages = average(bicycle_data)  # Create averages row
travel_one_hot = one_hot_encode(averages, 'Travel motives') # One-hot encode the 'Travel motives' column

model, t_test = linear_regression(travel_one_hot, 'Average travel distance (km) per day', 'Travel motives')

print(t_test.summary())  # Print the t-test summary

visualisations(travel_one_hot, model) # Create visualisations and save them to the output folder

# Save the processed data to a CSV file
travel_one_hot.to_csv(OUTPUT_FOLDER_CSV / 'processed_travel_data.csv', index=False) # Save the processed data to a CSV file

# Save the regression model summary to a CSV file
model_summary_df = pd.DataFrame(model.summary().tables[1].data[1:], columns=model.summary().tables[1].data[0])
model_summary_df.to_csv(OUTPUT_FOLDER_VIS / 'regression_model_summary.html', index=False) # Save the regression model summary to a CSV file
print(model_summary_df)

print("The Research question being answered is: What travel motive do cyclists in the Netherlands cycle the longest duration for, and how does age influence this?")
print("The hypothesis being tested is: There is a significant difference in average cycling distance between cycling for leisure and cycling to commute to work.")
print("As shown in the visualisation, the average travel distance for leisure is slightly higher than for commuting to work. However, the difference is not statistically significant. The coefficients only differ by 0.01, and both coefficients are positive. This indicates that the average travel distance for leisure is slightly higher than for commuting to work, but the difference is not statistically significant. They also both have P value > 0.05.") 