import json
import pandas as pd


df = pd.read_csv(r"C:\Users\user\Desktop\새 폴더\semart_test_cleaned_final.csv", encoding = 'cp949')

def convert_to_alpaca_format(df):
    alpaca_data = []
    for _, row in df.iterrows():
        entry = {
            "instruction": f"Describe the artwork {row['TITLE']} by {row['AUTHOR']}.",
            "input": row['DESCRIPTION'],
            "output": row['TITLE']
        }
        alpaca_data.append(entry)
    return alpaca_data

alpaca_data = convert_to_alpaca_format(df)

# JSON 파일로 저장
with open('csv_alpaca_data.json', 'w') as f:
    json.dump(alpaca_data, f, indent=4)
