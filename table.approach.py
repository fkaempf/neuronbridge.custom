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

# Load Google Sheets data
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
cred_path = '/Users/fkampf/My Drive/phd/Jefferis Lab/google.cloud/general-461415-aeee4ddcc160.json'
creds = Credentials.from_service_account_file(cred_path, scopes=scope)
client = gspread.authorize(creds)
sheet1 = client.open("13398.matching.sheet").worksheet('full').get_all_records()
sheet2 = client.open("12383.matching.sheet").worksheet('full').get_all_records()
sheet3 = client.open("12425.matching.sheet").worksheet('full').get_all_records()
sheet4 = client.open("13416.matching.sheet").worksheet('full').get_all_records()
sheet5 = client.open("11055.matching.sheet").worksheet('full').get_all_records()
df1 = pd.DataFrame(sheet1)
df2 = pd.DataFrame(sheet2)
df3 = pd.DataFrame(sheet3)
df4 = pd.DataFrame(sheet4)
df5 = pd.DataFrame(sheet5)
df_vAB3 = pd.concat([df1, df2, df3], ignore_index=True)
df_PPN1 = pd.concat([df4, df5], ignore_index=True)
df_vAB3['neuron'] = 'vAB3'
df_PPN1['neuron'] = 'PPN1'
df = pd.concat([df_PPN1,df_vAB3], ignore_index=True)
df_grouped = df.groupby(['neuron', 'Line Name']).mean(numeric_only=True).reset_index()

df_wide = df_grouped.pivot_table(
    index='Line Name',
    columns='neuron',
    values=['Score','Matched Pixels']
).reset_index()



import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Replace 0 with NaN to avoid division by zero
score_denominator = df_wide[('Score', 'vAB3')].replace(0, np.nan)
matched_denominator = df_wide[('Matched Pixels', 'vAB3')].replace(0, np.nan)

# Calculate ratios
df_wide['Score Ratio'] = df_wide[('Score', 'PPN1')] / score_denominator
df_wide['Matched Ratio'] = df_wide[('Matched Pixels', 'PPN1')] / matched_denominator

# Quantile thresholds
score_ppn1_q90 = df_wide[('Score', 'PPN1')].quantile(0.9)
score_vab3_q90 = df_wide[('Score', 'vAB3')].quantile(0.9)
ratio_q90 = df_wide['Score Ratio'].quantile(0.9)
ratio_q10 = df_wide['Score Ratio'].quantile(0.1)

# Select lines to label
label_mask = (
    ((df_wide[('Score', 'PPN1')] > score_ppn1_q90) | (df_wide[('Score', 'vAB3')] > score_vab3_q90)) &
    ((df_wide['Score Ratio'] > ratio_q90) | (df_wide['Score Ratio'] < ratio_q10))
)

# Add labels (line names) where mask is True
df_wide['Label'] = np.where(label_mask, df_wide['Line Name'], np.nan)

# Create scatterplot
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(
    data=df_wide,
    x=('Score', 'vAB3'),
    y=('Score', 'PPN1')
)

# Add identity line (y = x)
min_val = np.nanmin([df_wide[('Score', 'vAB3')].min(), df_wide[('Score', 'PPN1')].min()])
max_val = np.nanmax([df_wide[('Score', 'vAB3')].max(), df_wide[('Score', 'PPN1')].max()])
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', label='y = x')

# Annotate labeled points
for _, row in df_wide[df_wide['Label'].notna()].iterrows():
    ax.text(
        row[('Score', 'vAB3')]*1.05,
        row[('Score', 'PPN1')]*0.99,
        row['Label'][0],
        fontsize=6
    )

# Finalize plot
plt.xlabel('vAB3 Score')
plt.ylabel('PPN1 Score')
plt.title('vAB3 vs PPN1 Scores with Ratio-Based Highlighting')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Select lines to label
label_mask = (
    ((df_wide[('Matched Pixels', 'PPN1')] > score_ppn1_q90) | (df_wide[('Matched Pixels', 'vAB3')] > score_vab3_q90)) &
    ((df_wide['Matched Ratio'] > ratio_q90) | (df_wide['Matched Ratio'] < ratio_q10))
)

# Add labels (line names) where mask is True
df_wide['Label'] = np.where(label_mask, df_wide['Line Name'], np.nan)

# Create scatterplot
plt.figure(figsize=(10, 8))
ax = sns.scatterplot(
    data=df_wide,
    x=('Matched Pixels', 'vAB3'),
    y=('Matched Pixels', 'PPN1')
)

# Add identity line (y = x)
min_val = np.nanmin([df_wide[('Matched Pixels', 'vAB3')].min(), df_wide[('Matched Pixels', 'PPN1')].min()])
max_val = np.nanmax([df_wide[('Matched Pixels', 'vAB3')].max(), df_wide[('Matched Pixels', 'PPN1')].max()])
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', label='y = x')

# Annotate labeled points
for _, row in df_wide[df_wide['Label'].notna()].iterrows():
    ax.text(
        row[('Matched Pixels', 'vAB3')]*1.05,
        row[('Matched Pixels', 'PPN1')]*0.99,
        row['Label'][0],
        fontsize=6
    )

# Finalize plot
plt.xlabel('vAB3 Matched Pixels')
plt.ylabel('PPN1 Matched Pixels')
plt.title('vAB3 vs PPN1 Matched Pixels with Ratio-Based Highlighting')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()