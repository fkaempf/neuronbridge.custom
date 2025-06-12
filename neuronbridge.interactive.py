"""
neuronbridge.interactive.py

Comprehensive workflow for fetching, analyzing, and visualizing neuron matching data from the Janelia NeuronBridge project.
Compares two neuron cell types (vAB3 and PPN1) using their matching scores and matching pixels, highlights outliers,
and generates both static (matplotlib) and interactive (Plotly) scatterplots. Interactive plots allow
clicking on points to open associated CDM images or download Visually Lossless Stacks (VLS).

-------------------------------------------------------------------------------
Workflow Overview:
1. Fetch metadata for vAB3 and PPN1 neuron IDs from NeuronBridge S3.
2. Filter for FlyEM_MANC_v1.0 library and fetch corresponding CDS results.
3. Build a DataFrame with normalized scores and matching pixels for each cell type.
4. Calculate ratios and quantile thresholds to highlight interesting lines.
5. Create static scatterplots with matplotlib, labeling outliers.
6. Create interactive Plotly scatterplots (scores and matching pixels) with clickable points linking to CDM images or VLS stacks.

-------------------------------------------------------------------------------
Dependencies:
- pandas, numpy, requests, seaborn, matplotlib, plotly, tqdm

Usage:
    python neuronbridge.interactive.py

Outputs:
- Static scatterplots (matplotlib)
- Interactive HTML scatterplots:
    - clickable_brain_plot.html
    - clickable_vnc_plot.html
    - clickable_brain_plot_matchingpixels.html
    - clickable_vnc_plot_matchingpixels.html
    - clickable_vls_plot.html
    - clickable_vls_plot_matchingpixels.html

-------------------------------------------------------------------------------
Data sources:
- Janelia NeuronBridge S3: https://neuronbridge.janelia.org/
- vAB3 and PPN1 cell types, FlyEM_MANC_v1.0 library

-------------------------------------------------------------------------------
Author: Florian Kämpf
Date: 2025-06-11
-------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# For each neuron, fetch color depth search (CDS) results and build links to CDM and VLS images
link_df2 = pd.DataFrame()
for _, item in tqdm(link_df.iterrows(), total=link_df.shape[0], desc="Fetching CDS results"):
    path = f"https://janelia-neuronbridge-data-prod.s3.amazonaws.com/{version}/metadata/cdsresults/{item['id']}.json"
    response = requests.get(path)
    temp = pd.json_normalize(response.json()['results'])
    temp['query_id'] = item['id']
    temp['cell_type'] = item['cell_type']
    temp['link2cdm'] = 'https://s3.amazonaws.com/janelia-flylight-color-depth/' + temp['image.files.CDM']
    temp['link2vls'] = 'https://s3.amazonaws.com/janelia-flylight-imagery/' + temp['image.files.VisuallyLosslessStack']
    link_df2 = pd.concat([link_df2, temp], ignore_index=True)

# --- BUILD DATAFRAME OF UNIQUE PUBLISHED NAMES AND THEIR CDM/VLS LINKS ---
# Used for merging with main analysis DataFrame for interactive plots
linkdf = link_df2.loc[:, ['image.publishedName', 'link2cdm','link2vls', 'image.id']].drop_duplicates(subset='image.publishedName', keep='first')

# --- FETCH GAL4 AND CDM LINKS FOR BRAIN/VNC, MALE/FEMALE IN PARALLEL ---
def fetch_gal4_links(published_name):
    """
    Fetches additional Gal4 and CDM links for a given published_name from NeuronBridge S3.
    Returns a tuple of links for male/female, brain/VNC, Gal4/CDM.
    """
    import time

    def safe_extract(df, gender, area, colname, prefix):
        try:
            values = (
                df.loc[
                    (df.get('gender') == gender) & (df.get('anatomicalArea') == area),
                    colname
                ]
                .dropna()
                .unique()
                .tolist()
            )
            return f'{prefix}{values[0]}' if values else None
        except Exception:
            return None

    url = f'https://janelia-neuronbridge-data-prod.s3.amazonaws.com/v3_4_1/metadata/by_line/{published_name}.json'
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            break
        except requests.exceptions.ConnectionError as e:
            if 'Connection reset by peer' in str(e) and attempt < 2:
                time.sleep(1)
                continue
            else:
                return (published_name,) + (None,) * 8
        except Exception:
            return (published_name,) + (None,) * 8

    try:
        temp = pd.json_normalize(response.json().get('results', []))
    except Exception:
        return (published_name,) + (None,) * 8

    # Define fields and URLs
    fields = [
        ('m', 'Brain', 'files.Gal4Expression', 'https://s3.amazonaws.com/janelia-flylight-imagery/'),
        ('m', 'VNC', 'files.Gal4Expression', 'https://s3.amazonaws.com/janelia-flylight-imagery/'),
        ('f', 'Brain', 'files.Gal4Expression', 'https://s3.amazonaws.com/janelia-flylight-imagery/'),
        ('f', 'VNC', 'files.Gal4Expression', 'https://s3.amazonaws.com/janelia-flylight-imagery/'),
        ('m', 'Brain', 'files.CDM', 'https://s3.amazonaws.com/janelia-flylight-color-depth/'),
        ('m', 'VNC', 'files.CDM', 'https://s3.amazonaws.com/janelia-flylight-color-depth/'),
        ('f', 'Brain', 'files.CDM', 'https://s3.amazonaws.com/janelia-flylight-color-depth/'),
        ('f', 'VNC', 'files.CDM', 'https://s3.amazonaws.com/janelia-flylight-color-depth/'),
    ]

    links = [safe_extract(temp, gender, area, col, prefix) for gender, area, col, prefix in fields]

    return (published_name, *links)

# Parallel fetching with threads for all published names
results = []
with ThreadPoolExecutor(max_workers=50) as executor:
    futures = {executor.submit(fetch_gal4_links, name): name for name in linkdf['image.publishedName']}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Gal4 links"):
        results.append(future.result())

# Update DataFrame with results efficiently
for name, link2gal4_m_brain, link2gal4_m_vnc, link2gal4_f_brain, link2gal4_f_vnc, link2cdm_m_brain, link2cdm_m_vnc, link2cdm_f_brain, link2cdm_f_vnc in results:
    linkdf.loc[linkdf['image.publishedName'] == name, 'link2gal4_m_brain'] = link2gal4_m_brain
    linkdf.loc[linkdf['image.publishedName'] == name, 'link2gal4_m_vnc'] = link2gal4_m_vnc
    linkdf.loc[linkdf['image.publishedName'] == name, 'link2gal4_f_brain'] = link2gal4_f_brain
    linkdf.loc[linkdf['image.publishedName'] == name, 'link2gal4_f_vnc'] = link2gal4_f_vnc
    linkdf.loc[linkdf['image.publishedName'] == name, 'link2cdm_m_brain'] = link2cdm_m_brain
    linkdf.loc[linkdf['image.publishedName'] == name, 'link2cdm_m_vnc'] = link2cdm_m_vnc
    linkdf.loc[linkdf['image.publishedName'] == name, 'link2cdm_f_brain'] = link2cdm_f_brain
    linkdf.loc[linkdf['image.publishedName'] == name, 'link2cdm_f_vnc'] = link2cdm_f_vnc

# --- COALESCE GAL4 AND CDM LINKS FOR BRAIN/VNC ---
# Use the first available link for each area (brain/vnc), prioritizing male Gal4, then male CDM, then female Gal4, then female CDM
linkdf['combined_brain'] = linkdf['link2gal4_m_brain'].combine_first(linkdf['link2cdm_m_brain']).combine_first(linkdf['link2gal4_f_brain']).combine_first(linkdf['link2cdm_f_brain'])
linkdf['combined_vnc'] = linkdf['link2gal4_m_vnc'].combine_first(linkdf['link2cdm_m_vnc']).combine_first(linkdf['link2gal4_f_vnc']).combine_first(linkdf['link2cdm_f_vnc'])
linkdf = linkdf.drop(columns=[
    'link2gal4_m_brain', 'link2gal4_m_vnc', 'link2gal4_f_brain', 'link2gal4_f_vnc',
    'link2cdm_m_brain', 'link2cdm_m_vnc', 'link2cdm_f_brain', 'link2cdm_f_vnc'
])

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
    # Place label to the right of the dot, with a horizontal offset for clarity
    ax.text(
        row[('normalizedScore', 'vAB3')] + 0.03 * (max_val - min_val),  # 3% of axis range to the right
        row[('normalizedScore', 'PPN1')],
        str(row['image.publishedName'].item()),  # Only the name
        fontsize=7,
        ha='left',   # align label left to the offset position
        va='center'  # center vertically with the dot
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
    # Place label to the right of the dot, with a horizontal offset for clarity
    ax.text(
        row[('matchingPixels', 'vAB3')] + 0.03 * (max_val - min_val),  # 3% of axis range to the right
        row[('matchingPixels', 'PPN1')],
        str(row['image.publishedName'].item()),  # Only the name
        fontsize=7,
        ha='left',   # align label left to the offset position
        va='center'  # center vertically with the dot
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
    """
    Flatten MultiIndex columns for easier access.
    Converts tuples to underscore-joined strings.
    """
    if isinstance(col, tuple):
        return '_'.join([str(c) for c in col if c])
    return col

# --- GENERIC INTERACTIVE PLOT FUNCTION ---
def make_interactive_plot(
    df, x, y, label_col, link_col, hover_col, plot_title, x_label, y_label, ratio_label, color_label, file_name
):
    """
    Create an interactive Plotly scatterplot with clickable points.
    - Points open the associated CDM, VLS, or Gal4/CDM link in a new tab.
    - Outliers are colored/labeled.
    - Isometric axes for direct comparison.
    - Output is saved as an HTML file.
    Args:
        df: DataFrame to plot
        x, y: column names for axes
        label_col: column for coloring/labeling outliers
        link_col: column with URLs for clickable points
        hover_col: column to show in hover tooltip
        plot_title: plot title
        x_label, y_label: axis labels
        ratio_label: label for ratio (for legend)
        color_label: label for color legend
        file_name: output HTML file name
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
    - Merges with link DataFrame for CDM/VLS/Gal4/CDM URLs
    Args:
        df_wide: main wide-format DataFrame
        label_mask: boolean mask for outlier labeling
        label_col: column name for label
        linkdf: DataFrame with publishedName and links
        link_field: which link field to use for clickable points
    Returns:
        DataFrame ready for plotting
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

# --- INTERACTIVE PLOTS (CDM/CDM/Gal4 and VLS, scores and matching pixels) ---

# 1. Interactive plot: normalized scores, brain (Gal4/CDM) links
df_plot = prepare_plot_df(df_wide, label_mask, 'Label', linkdf, 'combined_brain')
make_interactive_plot(
    df_plot,
    x='normalizedScore_vAB3',
    y='normalizedScore_PPN1',
    label_col='Label',
    link_col='combined_brain',
    hover_col='combined_brain',
    plot_title='vAB3 vs PPN1 Scores with Ratio-Based Highlighting (Click: View CDM/Brain)',
    x_label='vAB3 normalizedScore',
    y_label='PPN1 normalizedScore',
    ratio_label='normalizedScore Ratio',
    color_label='Label',
    file_name='clickable_brain_plot.html'
)

# 2. Interactive plot: normalized scores, VNC (Gal4/CDM) links
make_interactive_plot(
    df_plot,
    x='normalizedScore_vAB3',
    y='normalizedScore_PPN1',
    label_col='Label',
    link_col='combined_vnc',
    hover_col='combined_vnc',
    plot_title='vAB3 vs PPN1 Scores with Ratio-Based Highlighting (Click: View CDM/VNC)',
    x_label='vAB3 normalizedScore',
    y_label='PPN1 normalizedScore',
    ratio_label='normalizedScore Ratio',
    color_label='Label',
    file_name='clickable_vnc_plot.html'
)

# 3. Interactive plot: matching pixels, brain (Gal4/CDM) links
df_plot_mp = prepare_plot_df(df_wide, mp_label_mask, 'Label MP', linkdf, 'combined_brain')
make_interactive_plot(
    df_plot_mp,
    x='matchingPixels_vAB3',
    y='matchingPixels_PPN1',
    label_col='Label MP',
    link_col='combined_brain',
    hover_col='combined_brain',
    plot_title='vAB3 vs PPN1 Matching Pixels with Ratio-Based Highlighting (Click: View CDM/Brain)',
    x_label='vAB3 matchingPixels',
    y_label='PPN1 matchingPixels',
    ratio_label='matchingPixels Ratio',
    color_label='Label MP',
    file_name='clickable_brain_plot_matchingpixels.html'
)

# 4. Interactive plot: matching pixels, VNC (Gal4/CDM) links
df_plot_mp = prepare_plot_df(df_wide, mp_label_mask, 'Label MP', linkdf, 'combined_vnc')
make_interactive_plot(
    df_plot_mp,
    x='matchingPixels_vAB3',
    y='matchingPixels_PPN1',
    label_col='Label MP',
    link_col='combined_vnc',
    hover_col='combined_vnc',
    plot_title='vAB3 vs PPN1 Matching Pixels with Ratio-Based Highlighting (Click: View CDM/VNC)',
    x_label='vAB3 matchingPixels',
    y_label='PPN1 matchingPixels',
    ratio_label='matchingPixels Ratio',
    color_label='Label MP',
    file_name='clickable_vnc_plot_matchingpixels.html'
)

# 5. Interactive plot: normalized scores, VLS links
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

# 6. Interactive plot: matching pixels, VLS links
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
