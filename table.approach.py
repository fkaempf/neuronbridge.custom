"""
table.approach.py

This script fetches, processes, and visualizes neuron matching data from the Janelia NeuronBridge project.
It compares two neuron cell types (vAB3 and PPN1) using their matching scores and matching pixels, highlights outliers,
and generates both static (matplotlib) and interactive (Plotly) scatterplots. The interactive plots allow
clicking on points to open associated MIP images.

Main Steps:
1. Fetch metadata for vAB3 and PPN1 neuron IDs from NeuronBridge S3.
2. Filter for FlyEM_MANC_v1.0 library and fetch corresponding CDS results.
3. Build a DataFrame with normalized scores and matching pixels for each cell type.
4. Calculate ratios and quantile thresholds to highlight interesting lines.
5. Create static scatterplots with matplotlib, labeling outliers.
6. Create interactive Plotly scatterplots (scores and matching pixels) with clickable points linking to MIP images.

Dependencies:
- pandas, numpy, requests, seaborn, matplotlib, plotly, tqdm, gspread, google-auth

Usage:
    python table.approach.py

Outputs:
- Static scatterplots (matplotlib)
- Interactive HTML scatterplots (clickable_mip_plot.html, clickable_mip_plot_matchingpixels.html)
"""

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm import tqdm

# --- PARAMETERS ---
vAB3_manc_ids = [13398, 12383, 12425]
PPN1_manc_ids = [13416, 11055]
version = "v3_4_0"
dataset = 'by_body'

# --- FETCH METADATA FOR NEURON IDS ---
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
link_df = link_df.loc[link_df['libraryName'] == 'FlyEM_MANC_v1.0', ['id', 'cell_type']]

# --- FETCH CDS RESULTS FOR EACH NEURON ---
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

# --- BUILD DATAFRAME OF UNIQUE PUBLISHED NAMES AND THEIR MIP LINKS ---
linkdf = link_df2.loc[:, ['image.publishedName', 'link2mip', 'image.id']].drop_duplicates(subset='image.publishedName', keep='first')

# --- GROUP BY CELL TYPE AND PUBLISHED NAME, AGGREGATE BY MEAN ---
df_grouped = link_df2.groupby(['cell_type', 'image.publishedName']).mean(numeric_only=True).reset_index()

# --- PIVOT TO WIDE FORMAT FOR COMPARISON ---
df_wide = df_grouped.pivot_table(
    index='image.publishedName',
    columns='cell_type',
    values=['normalizedScore', 'matchingPixels']
).reset_index()

# --- CALCULATE RATIOS AND QUANTILE THRESHOLDS ---
score_denominator = df_wide[('normalizedScore', 'vAB3')].replace(0, np.nan)
matched_denominator = df_wide[('matchingPixels', 'vAB3')].replace(0, np.nan)

df_wide['normalizedScore Ratio'] = df_wide[('normalizedScore', 'PPN1')] / score_denominator
df_wide['matchingPixels Ratio'] = df_wide[('matchingPixels', 'PPN1')] / matched_denominator

score_ppn1_q90 = df_wide[('normalizedScore', 'PPN1')].quantile(0.9)
score_vab3_q90 = df_wide[('normalizedScore', 'vAB3')].quantile(0.9)
ratio_q90 = df_wide['normalizedScore Ratio'].quantile(0.9)
ratio_q10 = df_wide['normalizedScore Ratio'].quantile(0.1)

mp_ppn1_q90 = df_wide[('matchingPixels', 'PPN1')].quantile(0.9)
mp_vab3_q90 = df_wide[('matchingPixels', 'vAB3')].quantile(0.9)
mp_ratio_q90 = df_wide['matchingPixels Ratio'].quantile(0.9)
mp_ratio_q10 = df_wide['matchingPixels Ratio'].quantile(0.1)

# --- SELECT LINES TO LABEL BASED ON QUANTILE THRESHOLDS ---
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
    if isinstance(col, tuple):
        return '_'.join([str(c) for c in col if c])
    return col

# --- INTERACTIVE PLOT: NORMALIZED SCORE ---
df_plot = df_wide.copy()
df_plot['Label'] = np.where(label_mask, 1, 0)
df_plot.loc[df_plot['image.publishedName'] == 'SS56947', 'Label'] = '2'
df_plot.columns = [clean_col(col) for col in df_wide.columns]
df_plot.fillna(0, inplace=True)
df_plot = pd.merge(
    df_plot,
    linkdf,
    on='image.publishedName',
    how='left'
).sort_values(by=['normalizedScore_vAB3', 'normalizedScore_PPN1'], ascending=False).reset_index(drop=True)

fig = px.scatter(
    df_plot,
    x='normalizedScore_vAB3',
    y='normalizedScore_PPN1',
    hover_name='image.publishedName',
    hover_data={'link2mip': False},
    custom_data=['link2mip'],
    title='vAB3 vs PPN1 Scores with Ratio-Based Highlighting',
    labels={
        'normalizedScore_vAB3': 'vAB3 normalizedScore',
        'normalizedScore_PPN1': 'PPN1 normalizedScore',
        'normalizedScore Ratio': 'normalizedScore Ratio'
    },
    color='Label'
)
min_val = min(df_plot['normalizedScore_vAB3'].min(), df_plot['normalizedScore_PPN1'].min())
max_val = max(df_plot['normalizedScore_vAB3'].max(), df_plot['normalizedScore_PPN1'].max())
fig.add_shape(
    type='line',
    x0=min_val, y0=min_val,
    x1=max_val, y1=max_val,
    line=dict(dash='dash', color='gray')
)
fig.update_traces(marker=dict(size=8, opacity=0.7))
fig.update_layout(showlegend=False)
html_file = "clickable_mip_plot.html"
fig.write_html(html_file, include_plotlyjs='cdn', full_html=True)
with open(html_file, "a") as f:
    f.write("""
<script>
document.querySelectorAll('.plotly-graph-div').forEach(plot => {
    plot.on('plotly_click', function(data){
        var url = data.points[0].customdata[0];
        if (url) window.open(url, '_blank');
    });
});
</script>
""")
print(f"✅ Open '{html_file}' in a browser to test clickable points.")

# --- INTERACTIVE PLOT: MATCHING PIXELS ---
df_plot_mp = df_wide.copy()
df_plot_mp['Label MP'] = np.where(mp_label_mask, 1, 0)
df_plot_mp.loc[df_plot_mp['image.publishedName'] == 'SS56947', 'Label MP'] = '2'
df_plot_mp.columns = [clean_col(col) for col in df_wide.columns] + [
    col for col in df_plot_mp.columns if col not in df_wide.columns
]
df_plot_mp.fillna(0, inplace=True)
df_plot_mp = pd.merge(
    df_plot_mp,
    linkdf,
    on='image.publishedName',
    how='left'
).sort_values(by=['matchingPixels_vAB3', 'matchingPixels_PPN1'], ascending=False).reset_index(drop=True)

fig_mp = px.scatter(
    df_plot_mp,
    x='matchingPixels_vAB3',
    y='matchingPixels_PPN1',
    hover_name='image.publishedName',
    hover_data={'link2mip': False},
    custom_data=['link2mip'],
    title='vAB3 vs PPN1 Matching Pixels with Ratio-Based Highlighting',
    labels={
        'matchingPixels_vAB3': 'vAB3 matchingPixels',
        'matchingPixels_PPN1': 'PPN1 matchingPixels',
        'matchingPixels Ratio': 'matchingPixels Ratio'
    },
    color='Label MP'
)
min_val_mp = min(df_plot_mp['matchingPixels_vAB3'].min(), df_plot_mp['matchingPixels_PPN1'].min())
max_val_mp = max(df_plot_mp['matchingPixels_vAB3'].max(), df_plot_mp['matchingPixels_PPN1'].max())
fig_mp.add_shape(
    type='line',
    x0=min_val_mp, y0=min_val_mp,
    x1=max_val_mp, y1=max_val_mp,
    line=dict(dash='dash', color='gray')
)
fig_mp.update_traces(marker=dict(size=8, opacity=0.7))
fig_mp.update_layout(showlegend=False)
html_file_mp = "clickable_mip_plot_matchingpixels.html"
fig_mp.write_html(html_file_mp, include_plotlyjs='cdn', full_html=True)
with open(html_file_mp, "a") as f:
    f.write("""
<script>
document.querySelectorAll('.plotly-graph-div').forEach(plot => {
    plot.on('plotly_click', function(data){
        var url = data.points[0].customdata[0];
        if (url) window.open(url, '_blank');
    });
});
</script>
""")
print(f"✅ Open '{html_file_mp}' in a browser to test clickable points for matching pixels.")
