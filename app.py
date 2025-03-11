from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import requests
from io import StringIO, BytesIO


app = Flask(__name__)

# GitHub Raw Data Base URL
GITHUB_BASE_URL = "https://raw.githubusercontent.com/SamihaShahid/CTI_barplot_tool/main/data/"

# Function to load CSV files from GitHub
def load_data_from_github(filename):
    url = GITHUB_BASE_URL + filename
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad response (e.g., 404)
        return pd.read_csv(StringIO(response.text))
    except requests.exceptions.RequestException as e:
        print(f"Error loading {filename}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if loading fails

# Function to load Excel files from GitHub
def load_excel_from_github(filename):
    url = GITHUB_BASE_URL + filename
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad response (e.g., 404)
        return pd.read_excel(BytesIO(response.content), sheet_name="Sheet1")
    except requests.exceptions.RequestException as e:
        print(f"Error loading {filename}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if loading fails

# Load emissions and health data from GitHub
mm1 = load_excel_from_github("processed_ems_2020_rpt_grp.xlsx")
mm2 = load_excel_from_github("processed_ems_2010.xlsx")
mm3 = load_excel_from_github("processed_ems_2008.xlsx")
hh = load_data_from_github("MASTER_HEALTH_202503100840.csv")

#_____
hh = hh[['POL', 'InhalationCancerURF', 'InhalationChronicREL', 'AcuteREL', 'MolWtCorrection']]
hh = hh.rename(columns={'POL': 'pol'})

# Merge emission datasets
mm1 = mm1.merge(mm2[['SP_10', 'SA_10', 'A_10', 'M_10', 'O_10', 'N_10', 'pol']], on='pol', how='left')

# Aggregate emissions from 2008
mm3['M_08'] = mm3['MG_08'] + mm3['MD_08'] + mm3["MO_08"]
mm3['O_08'] = mm3['OD_08'] + mm3['OG_08']

mm1 = mm1.merge(mm3[['SP_08', 'SA_08', 'A_08', 'M_08', 'O_08', 'N_08', 'poln']], on='poln', how='left')
mm1 = mm1.merge(hh, on='pol', how='left')

# Define emission source columns
ems_col = ['SP', 'SA', 'O', 'M', 'N', 'A', 
           'SP_10', 'SA_10', 'A_10', 'M_10', 'O_10', 'N_10', 
           'SP_08', 'SA_08', 'A_08', 'M_08', 'O_08', 'N_08']

# Handle zero values in health factors to avoid division errors
mm1[['InhalationCancerURF', 'MolWtCorrection']] = mm1[['InhalationCancerURF', 'MolWtCorrection']].replace(0, np.nan)

# Compute Cancer Toxic-Weighted Emissions (TWE)
mm1_cancer_twe = mm1.copy()
mm1_cancer_twe[ems_col] = mm1[ems_col].mul(
    mm1['InhalationCancerURF'] * mm1['MolWtCorrection'] * 7700, axis=0
)

# Compute Chonic Toxic-Weighted Emissions (TWE)
# Make a copy of mm1 to store chronic toxic-weighted emissions
mm1_chronic_twe = mm1.copy()
# Handle zero values in "InhalationChronicREL" to prevent division errors
mm1_chronic_twe['InhalationChronicREL'] = mm1_chronic_twe['InhalationChronicREL'].replace(0, np.nan)
# Compute Chronic Toxic-Weighted Emissions (CHRONIC_TWE)
mm1_chronic_twe[ems_col] = mm1[ems_col].mul(
    (1 / 8760) * (1 / mm1['InhalationChronicREL']) * 150, axis=0
)

# Compute Chonic Toxic-Weighted Emissions (TWE)
# Make a copy of mm1 to store acute toxic-weighted emissions
mm1_acute_twe = mm1.copy()
# Handle zero values in "AcuteREL" to prevent division errors
mm1_acute_twe['AcuteREL'] = mm1_acute_twe['AcuteREL'].replace(0, np.nan)
# Compute Acute Toxic-Weighted Emissions (ACUTE_TWE)
mm1_acute_twe[ems_col] = mm1[ems_col].mul(
    (1 / 8760) * (1 / mm1['AcuteREL']) * 1500, axis=0
)


# Define source categories
x_label = ['Stationary\nPoint', 'Stationary\nAggregate', 'Areawide', 'Onroad\nMobile', 'Other\nOnroad', 'Natural']

# Get unique pollutants
pollutants = mm1['poln'].dropna().unique()

def generate_plot(selected_pollutant):
    """Generate a 2x2 grid of subplots for:
       1. Emissions,
       2. Cancer Toxic-Weighted Emissions (TWE),
       3. Chronic Toxic-Weighted Emissions (TWE),
       4. Acute Toxic-Weighted Emissions (TWE)
    for the selected pollutant.
    """
    # Check if pollutant exists in dataset
    if selected_pollutant not in mm1['poln'].values:
        return None

    # Find row index for the selected pollutant
    selected_row = mm1[mm1['poln'] == selected_pollutant].index[0]

    # Define bar width and spacing
    bar_width = 0.3         # Width of each bar (for a single year)
    category_spacing = 0.5  # Additional spacing between categories

    # X positions for categories (with consistent spacing)
    x = np.arange(len(x_label)) * (3 * bar_width + category_spacing)

    # Extract Y values for each data type, grouping by year:
    # For raw emissions:
    y_emis_2008 = mm1[['SP_08', 'SA_08', 'A_08', 'M_08', 'O_08', 'N_08']].iloc[selected_row]
    y_emis_2010 = mm1[['SP_10', 'SA_10', 'A_10', 'M_10', 'O_10', 'N_10']].iloc[selected_row]
    y_emis_2020 = mm1[['SP', 'SA', 'A', 'M', 'O', 'N']].iloc[selected_row]

    # For Cancer TWE:
    y_cancer_2008 = mm1_cancer_twe[['SP_08', 'SA_08', 'A_08', 'M_08', 'O_08', 'N_08']].iloc[selected_row]
    y_cancer_2010 = mm1_cancer_twe[['SP_10', 'SA_10', 'A_10', 'M_10', 'O_10', 'N_10']].iloc[selected_row]
    y_cancer_2020 = mm1_cancer_twe[['SP', 'SA', 'A', 'M', 'O', 'N']].iloc[selected_row]

    # For Chronic TWE:
    y_chronic_2008 = mm1_chronic_twe[['SP_08', 'SA_08', 'A_08', 'M_08', 'O_08', 'N_08']].iloc[selected_row]
    y_chronic_2010 = mm1_chronic_twe[['SP_10', 'SA_10', 'A_10', 'M_10', 'O_10', 'N_10']].iloc[selected_row]
    y_chronic_2020 = mm1_chronic_twe[['SP', 'SA', 'A', 'M', 'O', 'N']].iloc[selected_row]

    # For Acute TWE:
    y_acute_2008 = mm1_acute_twe[['SP_08', 'SA_08', 'A_08', 'M_08', 'O_08', 'N_08']].iloc[selected_row]
    y_acute_2010 = mm1_acute_twe[['SP_10', 'SA_10', 'A_10', 'M_10', 'O_10', 'N_10']].iloc[selected_row]
    y_acute_2020 = mm1_acute_twe[['SP', 'SA', 'A', 'M', 'O', 'N']].iloc[selected_row]

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    pal = sns.color_palette('pastel', 10)

    # Plot 1: Raw Emissions
    axs[0, 0].bar(x - bar_width, y_emis_2008, color=pal[4], width=bar_width, edgecolor='k', label='2008', zorder=3)
    axs[0, 0].bar(x, y_emis_2010, color=pal[2], width=bar_width, edgecolor='k', label='2010', zorder=3)
    axs[0, 0].bar(x + bar_width, y_emis_2020, color=pal[0], width=bar_width, edgecolor='k', label='2020', zorder=3)
    axs[0, 0].set_title(f"Emissions for {selected_pollutant}", fontsize=14)
    axs[0, 0].set_ylabel('Emissions (tons/year)', fontsize=12)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(['','','','','',''])
    axs[0, 0].legend()
    axs[0, 0].grid(linestyle='--', color='#EBE7E0')

    # Plot 2: Cancer Toxic-Weighted Emissions
    axs[0, 1].bar(x - bar_width, y_cancer_2008, color=pal[4], width=bar_width, edgecolor='k', label='2008', zorder=3)
    axs[0, 1].bar(x, y_cancer_2010, color=pal[2], width=bar_width, edgecolor='k', label='2010', zorder=3)
    axs[0, 1].bar(x + bar_width, y_cancer_2020, color=pal[0], width=bar_width, edgecolor='k', label='2020', zorder=3)
    axs[0, 1].set_title(f"Cancer TWE for {selected_pollutant}", fontsize=14)
    axs[0, 1].set_ylabel('Cancer TWE', fontsize=12)
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(['','','','','',''])
    axs[0, 1].legend()
    axs[0, 1].grid(linestyle='--', color='#EBE7E0')

    # Plot 3: Chronic Toxic-Weighted Emissions
    axs[1, 0].bar(x - bar_width, y_chronic_2008, color=pal[4], width=bar_width, edgecolor='k', label='2008', zorder=3)
    axs[1, 0].bar(x, y_chronic_2010, color=pal[2], width=bar_width, edgecolor='k', label='2010', zorder=3)
    axs[1, 0].bar(x + bar_width, y_chronic_2020, color=pal[0], width=bar_width, edgecolor='k', label='2020', zorder=3)
    axs[1, 0].set_title(f"Chronic TWE for {selected_pollutant}", fontsize=14)
    axs[1, 0].set_ylabel('Chronic TWE', fontsize=12)
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(x_label, rotation=45, ha="right")
    axs[1, 0].legend()
    axs[1, 0].grid(linestyle='--', color='#EBE7E0')

    # Plot 4: Acute Toxic-Weighted Emissions
    axs[1, 1].bar(x - bar_width, y_acute_2008, color=pal[4], width=bar_width, edgecolor='k', label='2008', zorder=3)
    axs[1, 1].bar(x, y_acute_2010, color=pal[2], width=bar_width, edgecolor='k', label='2010', zorder=3)
    axs[1, 1].bar(x + bar_width, y_acute_2020, color=pal[0], width=bar_width, edgecolor='k', label='2020', zorder=3)
    axs[1, 1].set_title(f"Acute TWE for {selected_pollutant}", fontsize=14)
    axs[1, 1].set_ylabel('Acute TWE', fontsize=12)
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(x_label, rotation=45, ha="right")
    axs[1, 1].legend()
    axs[1, 1].grid(linestyle='--', color='#EBE7E0')

    # Adjust layout to prevent overlapping elements
    plt.tight_layout()

    # Save the figure to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()

    # Convert the image to a base64 string for embedding in HTML
    encoded_img = base64.b64encode(img.getvalue()).decode()
    return encoded_img

@app.route("/", methods=["GET", "POST"])
def home():
    selected_pollutant = request.form.get("pollutant", pollutants[0])
    plot_img = generate_plot(selected_pollutant)
    
    return render_template("index.html", pollutants=pollutants, selected_pollutant=selected_pollutant, plot_img=plot_img)

if __name__ == "__main__":
    app.run(debug=True)
