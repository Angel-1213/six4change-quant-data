from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from itertools import product

# 1. CONFIGURATION
# Define all paths upfront
DATA_PATH = Path(
    r"..\csv tables\transport_per_motive.csv")
OUTPUT_FOLDER_CSV = Path(
    r"..\csv tables")
OUTPUT_FOLDER_VIS = Path(
    r"..\visualisations\C.a.2")

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
    
    # Translations for travel motives (you can add more as needed)
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


    return bicycle_data

# One hot encoding for categorical variables
def one_hot_encode(data, column1, column2):
   encoder = OneHotEncoder(sparse_output=False, drop=['Total', 'Total']) # Initialise the encoder

   one_hot_encoded = encoder.fit_transform(data[[column1, column2]]) # Fit the encder to the data and transform it

   one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(), index=data.index) # Create a DataFrame from the encoded data

   df_encoded = pd.concat([data, one_hot_df], axis=1).dropna() # Concatenate the original DataFrame with the one-hot encoded DataFrame

   df_encoded = df_encoded.drop(columns=[column1, column2]) # Drop the original column after encoding
   return df_encoded # Return the DataFrame with the one-hot encoded columns

# interaction between travel motives and age groups
def interaction_terms(data, col1, col2):
    travel_cols = [col for col in data.columns if col.startswith(col1)]
    age_cols = [col for col in data.columns if col.startswith(col2)]

    interaction_data = {}  # dictionary to hold new columns

    for t_col, a_col in product(travel_cols, age_cols):
        interaction_col = f'{t_col} * {a_col}'
        interaction_data[interaction_col] = (
            pd.to_numeric(data[t_col], errors='coerce') *
            pd.to_numeric(data[a_col], errors='coerce')
        )

    # Convert dict to DataFrame and concat all new columns at once
    interaction_df = pd.DataFrame(interaction_data, index=data.index)
    combined_df = pd.concat([data.copy(), interaction_df], axis=1)
    return combined_df

# 3. LINEAR REGRESSION
def linear_regression(data, y, X):
    # Define predictors (all one-hot columns starting with 'Travel motives_')
    X = data.filter(like=X)

    # Define response variable
    y = data[y]

    # Add constant to the model (for intercept)
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X).fit()
    return model

# 4. VISUALISATION
# Visualise the coefficients of the regression model
def visualisations(model):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.params.values[1:], y=model.params.index[1:], palette='viridis')
    plt.xlabel('Average Travel Distance (km)')
    plt.title('Average Travel Distance by Travel Motive × Age Group Interaction')
    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.savefig(OUTPUT_FOLDER_VIS / 'hyp_C.a.2.regression_coefficients.png')
    plt.show()
    return plt.show()

def plot_pvalues(model, significance_level=0.05):
    # Get p-values and coefficients
    pvals = model.pvalues[1:]  # Exclude the constant
    coefs = model.params[1:]

    # Convert to DataFrame for plotting
    df_pvals = pd.DataFrame({
        'Variable': pvals.index,
        'P-value': pvals.values,
        'Coefficient': coefs.values,
        'Significant': pvals.values < significance_level
    })

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_pvals, y='Variable', x='P-value', hue='Significant', dodge=False, palette={True: 'green', False: 'red'})
    plt.axvline(significance_level, color='black', linestyle='--', label=f'α = {significance_level}')
    plt.xlabel('P-value')
    plt.title('P-values of Regression Coefficients')
    plt.legend(title='Statistically Significant')
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER_VIS / 'hyp_C.a.2.regression_significance.png')
    plt.show()


# 5. EXECUTION

# Load and clean the data
bicycle_data = load_and_clean_data(DATA_PATH) 
travel_one_hot = one_hot_encode(bicycle_data, 'Travel motives', 'Age') # One-hot encode the 'Travel motives' column

# Create interaction terms between travel motives and age groups
interaction_df = interaction_terms(travel_one_hot, 'Travel motives', 'Age') # Create interaction terms between travel motives and age groups

# Perform linear regression
model = linear_regression(interaction_df, 'Average traveltime per tip (minutes)', 'Travel motives_')

model_summary_inter = pd.DataFrame(model.summary().tables[1].data[1:], columns=model.summary().tables[1].data[0])
model_summary_inter.to_csv(OUTPUT_FOLDER_VIS / 'regression_inter_model_summary.html', index=False) # Save the regression model summary to a CSV file

# Visualise the coefficients of the regression model
visualisation = visualisations(model) 

print("The Research question being answered is: What travel motive do cyclists in the Netherlands cycle the longest duration for, and how does age influence this?")
print("The hypothesis is: H C.a.2: The relationship between travel motive and duration of a cycling trip differs across age groups.")
print("While the time traveled for different travel motives does differ across age groups, the difference is not statistically significant. The only significant difference was between Age 6-12 for Leisure and the constsant (total).")