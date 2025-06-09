import pandas as pd 
import google.auth
import gspread 
import requests
import os
from google.oauth2.service_account import Credentials
from tqdm import tqdm
from google.oauth2.service_account import Credentials
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests



#requests package is used to fetch JSON data from a URL
vAB3_manc_ids = [13398, 12383, 12425]
PPN1_manc_ids = [13416, 11055]
link_df = pd.DataFrame()
version = "v3_4_0"
dataset = 'by_body'
for id,neuron_type in zip(np.concatenate([vAB3_manc_ids,PPN1_manc_ids]), ['vAB3']*len(vAB3_manc_ids) + ['PPN1']*len(PPN1_manc_ids)):
    path = f"https://janelia-neuronbridge-data-prod.s3.amazonaws.com/{version}/metadata/{dataset}/{id}.json?x-id=GetObject"
    response = requests.get(path)
    temp = pd.json_normalize(response.json()['results'])
    temp['cell_type'] = neuron_type
    link_df = pd.concat([link_df,temp], ignore_index=True)


link_df = link_df.loc[link_df['libraryName']=='FlyEM_MANC_v1.0',['id','cell_type']]

link_df2 = pd.DataFrame()
for i, item in link_df.iterrows():
    path = url = f"https://janelia-neuronbridge-data-prod.s3.amazonaws.com/{version}/metadata/cdsresults/{item['id']}.json"
    response = requests.get(path)
    temp = pd.json_normalize(response.json()['results'])
    temp['query_id'] = item['id']
    temp['cell_type'] = item['cell_type']
    temp['link2mip'] = 'https://s3.amazonaws.com/janelia-flylight-color-depth/' + temp['image.files.CDM']
    link_df2 = pd.concat([link_df2,temp], ignore_index=True)
    
linkdf = link_df2.loc[:,['image.publishedName','link2mip']].drop_duplicates(subset='image.publishedName',keep='first')



df_grouped = link_df2.groupby(['cell_type', 'image.publishedName']).mean(numeric_only=True).reset_index()

df_wide = df_grouped.pivot_table(
    index='image.publishedName',
    columns='cell_type',
    values=['normalizedScore','matchingPixels']
).reset_index()




import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Replace 0 with NaN to avoid division by zero
score_denominator = df_wide[('normalizedScore', 'vAB3')].replace(0, np.nan)
matched_denominator = df_wide[('matchingPixels', 'vAB3')].replace(0, np.nan)

# Calculate ratios
df_wide['normalizedScore Ratio'] = df_wide[('normalizedScore', 'PPN1')] / score_denominator
df_wide['matchingPixels Ratio'] = df_wide[('matchingPixels', 'PPN1')] / matched_denominator

# Quantile thresholds
score_ppn1_q90 = df_wide[('normalizedScore', 'PPN1')].quantile(0.9)
score_vab3_q90 = df_wide[('normalizedScore', 'vAB3')].quantile(0.9)
ratio_q90 = df_wide['normalizedScore Ratio'].quantile(0.9)
ratio_q10 = df_wide['normalizedScore Ratio'].quantile(0.1)

# Select lines to label
label_mask = (
    ((df_wide[('normalizedScore', 'PPN1')] > score_ppn1_q90) | (df_wide[('normalizedScore', 'vAB3')] > score_vab3_q90)) &
    ((df_wide['normalizedScore Ratio'] > ratio_q90) | (df_wide['normalizedScore Ratio'] < ratio_q10))
)

# Add labels (image.publishedNames) where mask is True
df_wide['Label'] = np.where(label_mask, df_wide['image.publishedName'], np.nan)

# Create scatterplot
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(
    data=df_wide,
    x=('normalizedScore', 'vAB3'),
    y=('normalizedScore', 'PPN1')
)

# Add identity line (y = x)
min_val = np.nanmin([df_wide[('normalizedScore', 'vAB3')].min(), df_wide[('normalizedScore', 'PPN1')].min()])
max_val = np.nanmax([df_wide[('normalizedScore', 'vAB3')].max(), df_wide[('normalizedScore', 'PPN1')].max()])
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', label='y = x')

# Annotate labeled points
for _, row in df_wide[df_wide['Label'].notna()].iterrows():
    ax.text(
        row[('normalizedScore', 'vAB3')]*1.05,
        row[('normalizedScore', 'PPN1')]*0.99,
        row['Label'][0],
        fontsize=6
    )

# Finalize plot
plt.xlabel('vAB3 normalizedScore')
plt.ylabel('PPN1 normalizedScore')
plt.title('vAB3 vs PPN1 Scores with Ratio-Based Highlighting')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


import plotly.express as px

# Flatten the multi-index columns if needed
df_plot = df_wide.copy()
df_plot['Label'] = np.where(label_mask, 1, 0)
df_plot.loc[df_plot['image.publishedName'] == 'SS56947','Label'] = '2'

def clean_col(col):
    if isinstance(col, tuple):
        return '_'.join([str(c) for c in col if c])  # only join non-empty parts
    return col

df_plot.columns = [clean_col(col) for col in df_wide.columns]
df_plot.fillna(0, inplace=True)

df_plot = pd.merge(
    df_plot,
    linkdf,
    on='image.publishedName',
    how='left'
).sort_values(by=['normalizedScore_vAB3','normalizedScore_PPN1'], ascending=False).reset_index(drop=True)


import plotly.express as px

fig = px.scatter(
    df_plot,
    x='normalizedScore_vAB3',
    y='normalizedScore_PPN1',
    hover_name='image.publishedName',
    hover_data={'link2mip': False},  # hide from hover
    custom_data=['link2mip'],        # still available for JS
    title='vAB3 vs PPN1 Scores with Ratio-Based Highlighting',
    labels={
        'normalizedScore_vAB3': 'vAB3 normalizedScore',
        'normalizedScore_PPN1': 'PPN1 normalizedScore',
        'normalizedScore Ratio\t': 'normalizedScore Ratio'
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

# Save HTML and append JS for clickable points
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