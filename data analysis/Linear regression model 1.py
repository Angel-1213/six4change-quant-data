from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. CONFIGURATION
# Define all paths upfront
DATA_PATH = Path(
    r"C:\Users\ThinkBook\Desktop\netherlands\six4change-quant-data\csv tables\mobility_by_region.csv")
OUTPUT_FOLDER_CSV = Path(
    r"C:\Users\ThinkBook\Desktop\netherlands\six4change-quant-data\csv tables")
OUTPUT_FOLDER_VIS = Path(
    r"C:\Users\ThinkBook\Desktop\netherlands\six4change-quant-data\visualisations")

# Ensure output folders exist
for folder in [OUTPUT_FOLDER_CSV, OUTPUT_FOLDER_VIS]:
    folder.mkdir(parents=True, exist_ok=True)


# 2. DATA PROCESSING

def load_and_clean_data(file_path):
    """Load and clean the raw data"""
    data = pd.read_csv(file_path)

    # Filter only 'Value' rows
    bicycle_data = data[data['Margins'] == 'Value'].copy()

    # Convert European number formats
    numeric_cols = [
        'Average distance per trip (passenger-kilometres)',
        'Average travel duration per trip (minutes)'
    ]

    for col in numeric_cols:
        bicycle_data[col] = (
            bicycle_data[col]
            .astype(str)
            .str.replace(',', '.')
            .astype(float)
        )

    # Extract year from Periods
    bicycle_data['Year'] = (
        bicycle_data['Periods']
        .astype(str)
        .str.extract(r'(\d+)')[0]
        .astype(int)
    )

    return bicycle_data


# 3. VISUALIZATION

def create_visualizations(data, output_folder):
    """Create and save all visualizations"""
    # Visualization 1: Distance trends
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x='Year',
        y='Average distance per trip (passenger-kilometres)',
        hue='Regions',
        style='Regions',
        markers=True,
        dashes=False
    )
    plt.title("Cycling Distance Trends by Region (2018-2023)")
    plt.ylabel("Cycling distance (km)")
    plt.xlabel("Year")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    distance_path = output_folder / "regional_cycling_distance_trends.png"
    plt.savefig(distance_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Distance visualization saved to: {distance_path}")

    # Visualization 2: Duration trends
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=data,
        x='Year',
        y='Average travel duration per trip (minutes)',
        hue='Regions',
        style='Regions',
        markers=True,
        dashes=False
    )
    plt.title("Cycling Duration Trends by Region (2018-2023)")
    plt.ylabel("Duration (minutes)")
    plt.xlabel("Year")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    duration_path = output_folder / "regional_cycling_duration_trends.png"
    plt.savefig(duration_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Duration visualization saved to: {duration_path}")


# 4. MAIN EXECUTION

if __name__ == "__main__":
    # Load and clean data
    bicycle_data = load_and_clean_data(DATA_PATH)

    # Save cleaned data
    cleaned_data_path = OUTPUT_FOLDER_CSV / "cleaned_mobility_data_by_region.csv"
    bicycle_data.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned data saved to: {cleaned_data_path}")

    # Create visualizations
    create_visualizations(bicycle_data, OUTPUT_FOLDER_VIS)

    print("Analysis completed successfully!")
