import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Load your data
df = pd.read_csv(
    r"C:\Users\ThinkBook\Desktop\Quantitative data analysis\csv tables\technology_adoption_by_age_and_gender.csv")

# 2. Create output folder (if it doesn't exist)
output_folder = r"C:\Users\ThinkBook\Desktop\Quantitative data analysis\visualisations"

# 3. Calculate correlations (select your target columns)
corr_matrix = df[['PEOU', 'Age', 'Gender']].corr()

# 4. Create styled visual matrix
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    fmt=".2f",
    linewidths=0.5,
    annot_kws={"size": 12}
)

# 5. Customize the plot
plt.title("Technology Adoption Correlations\n(PEOU vs Age & Gender)",
          pad=20, fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 6. Save high-quality image
heatmap_path = os.path.join(
    output_folder, "correlation_heatmap_age_and_education.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Visual correlation matrix saved to:\n{heatmap_path}")
