import pandas as pd 
import google.auth
import gspread 
import requests
import os
from google.oauth2.service_account import Credentials
from tqdm import tqdm
from google.oauth2.service_account import Credentials
import glob

# Load Google Sheets data
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
cred_path = '/Users/fkampf/My Drive/phd/Jefferis Lab/google.cloud/general-461415-aeee4ddcc160.json'
creds = Credentials.from_service_account_file(cred_path, scopes=scope)
client = gspread.authorize(creds)
sheet1 = client.open("13398.matching.sheet").worksheet('full').get_all_records()
sheet2 = client.open("12383.matching.sheet").worksheet('full').get_all_records()
sheet3 = client.open("12425.matching.sheet").worksheet('full').get_all_records()
df1 = pd.DataFrame(sheet1)
df2 = pd.DataFrame(sheet2)
df3 = pd.DataFrame(sheet3)
df = pd.concat([df1, df2, df3], ignore_index=True)


for i,item in df.iterrows():
    df.loc[i,'File Name'] = f"{item['Line Name']}-{item['Slide Code']}-*-{item['Sex']}-{item['Magnification']}-{item['Anatomical Area'].lower()}-{item['Alignment Space']}-CDM_{item['Channel']}-*.png"

target_dir = 'target'
def resolve_filename(pattern):
    full_pattern = os.path.join(target_dir, pattern)
    matches = glob.glob(full_pattern)
    if matches:
        return os.path.basename(matches[0])  # take the first match
    else:
        return None
    
df['Actual File Name'] = df['File Name'].apply(resolve_filename)


