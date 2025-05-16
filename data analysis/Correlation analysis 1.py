import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data from CSV
df = pd.read_csv(
    r"C:\Users\ThinkBook\Desktop\Quantitative data analysis\csv tables\technology_adoption_by_education.csv")

# Use the exact column name as shown in the output (case-sensitive)
ict_qual = df[df["Label"] ==
              "lack of ICT qualifications (training / education)"]

# Check if filtering worked
if ict_qual.empty:
    print("Error: No rows matched the filter. Check the exact label string.")
    print("Unique values in 'Label':", df["Label"].unique())
else:
    # Define numeric columns (use exact names from your CSV)
    numeric_cols = ["10-49", "50-249", "GE10", "GE250", "ICT", "Industry"]
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    # Calculate Pearson correlations
    correlations = {}
    for col in numeric_cols:
        valid_data = ict_qual[[col, "C10-S951_X_K"]].dropna()
        if len(valid_data) > 1:
            r, p_value = pearsonr(valid_data[col], valid_data["C10-S951_X_K"])
            correlations[col] = {"r": r, "p_value": p_value}

    # Save results
    output_folder = "visualisations"
    os.makedirs(output_folder, exist_ok=True)

    # Save cleaned data
    output_folder_1 = "csv tables"
    os.makedirs(output_folder_1, exist_ok=True)
    ict_qual.to_csv(os.path.join(
        output_folder_1, "cleaned_ict_qualification.csv"), index=False)

    # Generate and save correlation heatmap
    corr_matrix = ict_qual[numeric_cols].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation with Lack of ICT Qualifications")
    plt.savefig(os.path.join(output_folder, "correlation_heatmap_education.png"),
                bbox_inches="tight", dpi=300)
    plt.close()
