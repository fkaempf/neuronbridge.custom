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
sheet1 = client.open("13398.matching.sheet").worksheet('full').get_all_records()
sheet2 = client.open("12383.matching.sheet").worksheet('full').get_all_records()
sheet3 = client.open("12425.matching.sheet").worksheet('full').get_all_records()

# Example usage
df1 = pd.DataFrame(sheet1)
df2 = pd.DataFrame(sheet2)
df3 = pd.DataFrame(sheet3)
df = pd.concat([df1, df2, df3], ignore_index=True)


pre = 'https://s3.amazonaws.com/janelia-flylight-color-depth'

os.makedirs('target', exist_ok=True)
content_target = os.listdir('target')
for i, item in tqdm(df.iterrows(), total=df.shape[0]):
    error404 = True
    found = False
    for img_no in range(1, 10):
        img_no = f"{img_no:02d}"
        for Gal4 in ['Split_GAL4', 'GAL4']:
            url = f"{pre}/{item['Alignment Space']}/{item['Library'].replace(' ','_')}/searchable_neurons/pngs/{item['Line Name']}-{item['Slide Code']}-{Gal4}-{item['Sex']}-{item['Magnification']}-{item['Anatomical Area'].lower()}-{item['Alignment Space']}-CDM_{item['Channel']}-{img_no}.png"
            filename = url.split('/')[-1]
            if filename in content_target:
                error404 = False
                found = True
                break
            else:
                response = requests.get(url)
                if response.status_code == 200:
                    #print(f"Downloading {filename}")
                    with open(f'target/{filename}', 'wb') as f:
                        f.write(response.content)
                    error404 = False
                    found = True
                    break
        if found:
            break
    if error404:
        print(f"\033[91mFile not found: {filename}\033[0m")


    
 



    

