"""
neuronbridge.interactive.py

Comprehensive workflow for fetching, analyzing, and visualizing neuron matching data from the Janelia NeuronBridge project.
Compares two neuron cell types (vAB3 and PPN1) using their matching scores and matching pixels, highlights outliers,
and generates both static (matplotlib) and interactive (Plotly) scatterplots. Interactive plots allow
clicking on points to open associated MIP images or download Visually Lossless Stacks (VLS).

-------------------------------------------------------------------------------
Workflow Overview:
1. Fetch metadata for vAB3 and PPN1 neuron IDs from NeuronBridge S3.
2. Filter for FlyEM_MANC_v1.0 library and fetch corresponding CDS results.
3. Build a DataFrame with normalized scores and matching pixels for each cell type.
4. Calculate ratios and quantile thresholds to highlight interesting lines.
5. Create static scatterplots with matplotlib, labeling outliers.
6. Create interactive Plotly scatterplots (scores and matching pixels) with clickable points linking to MIP images or VLS stacks.

-------------------------------------------------------------------------------
Dependencies:
- pandas, numpy, requests, seaborn, matplotlib, plotly, tqdm

Usage:
    python neuronbridge.interactive.py

Outputs:
- Static scatterplots (matplotlib)
- Interactive HTML scatterplots:
    - clickable_mip_plot.html
    - clickable_vls_plot.html
    - clickable_mip_plot_matchingpixels.html
    - clickable_vls_plot_matchingpixels.html

-------------------------------------------------------------------------------
Data sources:
- Janelia NeuronBridge S3: https://neuronbridge.janelia.org/
- vAB3 and PPN1 cell types, FlyEM_MANC_v1.0 library

-------------------------------------------------------------------------------
Author: [Your Name]
Date: [Today's Date]
-------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm import tqdm

# --- PARAMETERS ---
# IDs for vAB3 and PPN1 neurons in the FlyEM_MANC_v1.0 library
vAB3_manc_ids = [13398, 12383, 12425]
PPN1_manc_ids = [13416, 11055]
version = "v3_4_0"
dataset = 'by_body'

# --- FETCH METADATA FOR NEURON IDS ---
# Download metadata for each neuron ID and assign cell type
link_df = pd.DataFrame()
for id, neuron_type in zip(
    np.concatenate([vAB3_manc_ids, PPN1_manc_ids]),
    ['vAB3'] * len(vAB3_manc_ids) + ['PPN1'] * len(PPN1_manc_ids)
):
    path = f"https://janelia-neuronbridge-data-prod.s3.amazonaws.com/{version}/metadata/{dataset}/{id}.json?x-id=GetObject"
    response = requests.get(path)
    temp = pd.json_normalize(response.json()['results'])
    temp['cell_type'] = neuron_type
    link_df = pd.concat([link_df, temp], ignore_index=True)

# --- FILTER FOR FLYEM_MANC_v1.0 LIBRARY ---
# Keep only entries from the FlyEM_MANC_v1.0 library
link_df = link_df.loc[link_df['libraryName'] == 'FlyEM_MANC_v1.0', ['id', 'cell_type']]

# --- FETCH CDS RESULTS FOR EACH NEURON ---
# For each neuron, fetch color depth search (CDS) results and build links to MIP and VLS images
link_df2 = pd.DataFrame()
for _, item in tqdm(link_df.iterrows(), total=link_df.shape[0], desc="Fetching CDS results"):
    path = f"https://janelia-neuronbridge-data-prod.s3.amazonaws.com/{version}/metadata/cdsresults/{item['id']}.json"
    response = requests.get(path)
    temp = pd.json_normalize(response.json()['results'])
    temp['query_id'] = item['id']
    temp['cell_type'] = item['cell_type']
    temp['link2mip'] = 'https://s3.amazonaws.com/janelia-flylight-color-depth/' + temp['image.files.CDM']
    temp['link2vls'] = 'https://s3.amazonaws.com/janelia-flylight-imagery/' + temp['image.files.VisuallyLosslessStack']
    link_df2 = pd.concat([link_df2, temp], ignore_index=True)

# --- BUILD DATAFRAME OF UNIQUE PUBLISHED NAMES AND THEIR MIP/VLS LINKS ---
# Used for merging with main analysis DataFrame for interactive plots
linkdf = link_df2.loc[:, ['image.publishedName', 'link2mip','link2vls', 'image.id']].drop_duplicates(subset='image.publishedName', keep='first')

# --- GROUP BY CELL TYPE AND PUBLISHED NAME, AGGREGATE BY MEAN ---
# For each cell type and published name, compute mean normalizedScore and matchingPixels
df_grouped = link_df2.groupby(['cell_type', 'image.publishedName']).mean(numeric_only=True).reset_index()

# --- PIVOT TO WIDE FORMAT FOR COMPARISON ---
# Create a wide DataFrame: each row is a publishedName, columns for vAB3 and PPN1 scores/pixels
df_wide = df_grouped.pivot_table(
    index='image.publishedName',
    columns='cell_type',
    values=['normalizedScore', 'matchingPixels']
).reset_index()

# --- CALCULATE RATIOS AND QUANTILE THRESHOLDS ---
# Compute ratios of PPN1/vAB3 for both normalizedScore and matchingPixels
score_denominator = df_wide[('normalizedScore', 'vAB3')].replace(0, np.nan)
matched_denominator = df_wide[('matchingPixels', 'vAB3')].replace(0, np.nan)

df_wide['normalizedScore Ratio'] = df_wide[('normalizedScore', 'PPN1')] / score_denominator
df_wide['matchingPixels Ratio'] = df_wide[('matchingPixels', 'PPN1')] / matched_denominator

# Quantile thresholds for highlighting outliers
score_ppn1_q90 = df_wide[('normalizedScore', 'PPN1')].quantile(0.9)
score_vab3_q90 = df_wide[('normalizedScore', 'vAB3')].quantile(0.9)
ratio_q90 = df_wide['normalizedScore Ratio'].quantile(0.9)
ratio_q10 = df_wide['normalizedScore Ratio'].quantile(0.1)

mp_ppn1_q90 = df_wide[('matchingPixels', 'PPN1')].quantile(0.9)
mp_vab3_q90 = df_wide[('matchingPixels', 'vAB3')].quantile(0.9)
mp_ratio_q90 = df_wide['matchingPixels Ratio'].quantile(0.9)
mp_ratio_q10 = df_wide['matchingPixels Ratio'].quantile(0.1)

# --- SELECT LINES TO LABEL BASED ON QUANTILE THRESHOLDS ---
# Label points that are outliers in either score or matching pixels
label_mask = (
    ((df_wide[('normalizedScore', 'PPN1')] > score_ppn1_q90) | (df_wide[('normalizedScore', 'vAB3')] > score_vab3_q90)) &
    ((df_wide['normalizedScore Ratio'] > ratio_q90) | (df_wide['normalizedScore Ratio'] < ratio_q10))
)
mp_label_mask = (
    ((df_wide[('matchingPixels', 'PPN1')] > mp_ppn1_q90) | (df_wide[('matchingPixels', 'vAB3')] > mp_vab3_q90)) &
    ((df_wide['matchingPixels Ratio'] > mp_ratio_q90) | (df_wide['matchingPixels Ratio'] < mp_ratio_q10))
)

# --- ADD LABELS WHERE MASK IS TRUE ---
df_wide['Label'] = np.where(label_mask, df_wide['image.publishedName'], np.nan)
df_wide['Label MP'] = np.where(mp_label_mask, df_wide['image.publishedName'], np.nan)

# --- STATIC SCATTERPLOT: NORMALIZED SCORE ---
# Plots vAB3 vs PPN1 normalized scores, labels outliers
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(
    data=df_wide,
    x=('normalizedScore', 'vAB3'),
    y=('normalizedScore', 'PPN1')
)
min_val = np.nanmin([df_wide[('normalizedScore', 'vAB3')].min(), df_wide[('normalizedScore', 'PPN1')].min()])
max_val = np.nanmax([df_wide[('normalizedScore', 'vAB3')].max(), df_wide[('normalizedScore', 'PPN1')].max()])
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', label='y = x')
for _, row in df_wide[df_wide['Label'].notna()].iterrows():
    ax.text(
        row[('normalizedScore', 'vAB3')]*1.05,
        row[('normalizedScore', 'PPN1')]*0.99,
        row['Label'][0],
        fontsize=6
    )
plt.xlabel('vAB3 normalizedScore')
plt.ylabel('PPN1 normalizedScore')
plt.title('vAB3 vs PPN1 Scores with Ratio-Based Highlighting')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- STATIC SCATTERPLOT: MATCHING PIXELS ---
# Plots vAB3 vs PPN1 matching pixels, labels outliers
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(
    data=df_wide,
    x=('matchingPixels', 'vAB3'),
    y=('matchingPixels', 'PPN1')
)
min_val = np.nanmin([df_wide[('matchingPixels', 'vAB3')].min(), df_wide[('matchingPixels', 'PPN1')].min()])
max_val = np.nanmax([df_wide[('matchingPixels', 'vAB3')].max(), df_wide[('matchingPixels', 'PPN1')].max()])
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', label='y = x')
for _, row in df_wide[df_wide['Label MP'].notna()].iterrows():
    ax.text(
        row[('matchingPixels', 'vAB3')]*1.05,
        row[('matchingPixels', 'PPN1')]*0.99,
        row['Label MP'][0],
        fontsize=6
    )
plt.xlabel('vAB3 matchingPixels')
plt.ylabel('PPN1 matchingPixels')
plt.title('vAB3 vs PPN1 Matching Pixels with Ratio-Based Highlighting')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- HELPER FUNCTION TO FLATTEN MULTIINDEX COLUMNS ---
def clean_col(col):
    """Flatten MultiIndex columns for easier access."""
    if isinstance(col, tuple):
        return '_'.join([str(c) for c in col if c])
    return col

# --- GENERIC INTERACTIVE PLOT FUNCTION ---
def make_interactive_plot(
    df, x, y, label_col, link_col, hover_col, plot_title, x_label, y_label, ratio_label, color_label, file_name
):
    """
    Create an interactive Plotly scatterplot with clickable points.
    - Points open the associated MIP or VLS link in a new tab.
    - Outliers are colored/labeled.
    - Isometric axes for direct comparison.
    - Output is saved as an HTML file.
    """
    fig = px.scatter(
        df,
        x=x,
        y=y,
        hover_name='image.publishedName',
        hover_data={hover_col: False},
        custom_data=[link_col],
        title=plot_title,
        labels={
            x: x_label,
            y: y_label,
            ratio_label: ratio_label
        },
        color=label_col
    )
    min_val = min(df[x].min(), df[y].min())
    max_val = max(df[x].max(), df[y].max())
    fig.add_shape(
        type='line',
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(dash='dash', color='gray')
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.write_html(file_name, include_plotlyjs='cdn', full_html=True)
    # Add JavaScript for clickable points
    with open(file_name, "a") as f:
        f.write(f"""
<script>
document.querySelectorAll('.plotly-graph-div').forEach(plot => {{
    plot.on('plotly_click', function(data){{
        var url = data.points[0].customdata[0];
        if (url) window.open(url, '_blank');
    }});
}});
</script>
""")
    print(f"✅ Open '{file_name}' in a browser to test clickable points.")

# --- DATAFRAME PREPARATION FUNCTION ---
def prepare_plot_df(df_wide, label_mask, label_col, linkdf, link_field):
    """
    Prepare DataFrame for plotting:
    - Flattens columns
    - Adds label column for outliers
    - Merges with link DataFrame for MIP/VLS URLs
    """
    df_plot = df_wide.copy()
    df_plot[label_col] = np.where(label_mask, 1, 0)
    # Optionally highlight a specific line (example: SS56947)
    df_plot.loc[df_plot['image.publishedName'] == 'SS56947', label_col] = '2'
    df_plot.columns = [clean_col(col) for col in df_wide.columns]
    df_plot.fillna(0, inplace=True)
    df_plot = pd.merge(
        df_plot,
        linkdf,
        on='image.publishedName',
        how='left'
    ).sort_values(by=[df_plot.columns[1], df_plot.columns[2]], ascending=False).reset_index(drop=True)
    return df_plot

# --- INTERACTIVE PLOTS (MIP and VLS, scores and matching pixels) ---

# 1. Interactive plot: normalized scores, MIP links
df_plot = prepare_plot_df(df_wide, label_mask, 'Label', linkdf, 'link2mip')
make_interactive_plot(
    df_plot,
    x='normalizedScore_vAB3',
    y='normalizedScore_PPN1',
    label_col='Label',
    link_col='link2mip',
    hover_col='link2mip',
    plot_title='vAB3 vs PPN1 Scores with Ratio-Based Highlighting (Click: View MIP)',
    x_label='vAB3 normalizedScore',
    y_label='PPN1 normalizedScore',
    ratio_label='normalizedScore Ratio',
    color_label='Label',
    file_name='clickable_mip_plot.html'
)

# 2. Interactive plot: normalized scores, VLS links
df_plot_vls = prepare_plot_df(df_wide, label_mask, 'Label', linkdf, 'link2vls')
make_interactive_plot(
    df_plot_vls,
    x='normalizedScore_vAB3',
    y='normalizedScore_PPN1',
    label_col='Label',
    link_col='link2vls',
    hover_col='link2vls',
    plot_title='vAB3 vs PPN1 Scores (Click: Download Stack)',
    x_label='vAB3 normalizedScore',
    y_label='PPN1 normalizedScore',
    ratio_label='normalizedScore Ratio',
    color_label='Label',
    file_name='clickable_vls_plot.html'
)

# 3. Interactive plot: matching pixels, MIP links
df_plot_mp = prepare_plot_df(df_wide, mp_label_mask, 'Label MP', linkdf, 'link2mip')
make_interactive_plot(
    df_plot_mp,
    x='matchingPixels_vAB3',
    y='matchingPixels_PPN1',
    label_col='Label MP',
    link_col='link2mip',
    hover_col='link2mip',
    plot_title='vAB3 vs PPN1 Matching Pixels with Ratio-Based Highlighting (Click: View MIP)',
    x_label='vAB3 matchingPixels',
    y_label='PPN1 matchingPixels',
    ratio_label='matchingPixels Ratio',
    color_label='Label MP',
    file_name='clickable_mip_plot_matchingpixels.html'
)

# 4. Interactive plot: matching pixels, VLS links
df_plot_mp_vls = prepare_plot_df(df_wide, mp_label_mask, 'Label MP', linkdf, 'link2vls')
make_interactive_plot(
    df_plot_mp_vls,
    x='matchingPixels_vAB3',
    y='matchingPixels_PPN1',
    label_col='Label MP',
    link_col='link2vls',
    hover_col='link2vls',
    plot_title='vAB3 vs PPN1 Matching Pixels (Click: Download Stack)',
    x_label='vAB3 matchingPixels',
    y_label='PPN1 matchingPixels',
    ratio_label='matchingPixels Ratio',
    color_label='Label MP',
    file_name='clickable_vls_plot_matchingpixels.html'
)

# --- END OF SCRIPT ---
