import pandas as pd 
import google.auth
import gspread 
import requests
import os
from google.oauth2.service_account import Credentials
from tqdm import tqdm
from google.oauth2.service_account import Credentials
scope = ["https://www.googleapis.com/auth/spreadsheets", 
         "https://www.googleapis.com/auth/drive"]
cred_path = '/Users/fkampf/My Drive/phd/Jefferis Lab/google.cloud/general-461415-aeee4ddcc160.json'

creds = Credentials.from_service_account_file(cred_path, scopes=scope)

# Authorize gspread with the credentials
client = gspread.authorize(creds)

# Open your spreadsheet
sheet = client.open("13398.matching.sheet").worksheet('full').get_all_records()

# Example usage
df = pd.DataFrame(sheet)

pre = 'https://s3.amazonaws.com/janelia-flylight-color-depth'

os.makedirs('target', exist_ok=True)
for i,item in tqdm(df.iterrows(),total=df.shape[0]):
    url = f'{pre}/{item['Alignment Space']}/{item['Library'].replace(' ','_')}/searchable_neurons/pngs/{item['Line Name']}-{item['Slide Code']}-Split_GAL4-{item['Sex']}-{item['Magnification']}-{item['Anatomical Area'].lower()}-{item['Alignment Space']}-CDM_1-01.png'
    
    if not f'{url.split('/')[-1]}' in os.listdir('target'):
        response = requests.get(url)
        if response.status_code == 200:
            with open(f'target/{url.split('/')[-1]}', 'wb') as f:
                f.write(response.content)
    
        else:
            url = f'{pre}/{item['Alignment Space']}/{item['Library'].replace(' ','_')}/searchable_neurons/pngs/{item['Line Name']}-{item['Slide Code']}-GAL4-{item['Sex']}-{item['Magnification']}-{item['Anatomical Area'].lower()}-{item['Alignment Space']}-CDM_1-01.png'
            response = requests.get(url)

    else:
        print(f'{url.split("/")[-1]} already exists, skipping download.')