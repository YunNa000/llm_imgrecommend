import json
import os

def convert_to_alpaca_format(aqua_data):
    alpaca_data = []
    for item in aqua_data:
        instruction = item["question"]
        input_text = ""  # AQUA 데이터셋은 추가 입력이 없는 경우가 많음
        output = item["answer"]
        alpaca_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output
        })
    return alpaca_data

# 데이터 파일 경로 - 절대 경로로 수정
train_file = 'D:/llm_AQUA_Alpaca/train.json'
val_file = 'D:/llm_AQUA_Alpaca/val.json'
test_file = 'D:/llm_AQUA_Alpaca/test.json'

# 데이터 로드
def load_data(file_path):
    if os.path.exists(file_path):
        print(f"Loading {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} entries from {file_path}")
            return data
    else:
        print(f"File {file_path} not found")
        return []

# AQUA 데이터셋 로드
aqua_train = load_data(train_file)
aqua_val = load_data(val_file)
aqua_test = load_data(test_file)

# 변환된 데이터 초기화
alpaca_train = []
alpaca_val = []
alpaca_test = []

# 데이터가 비어있지 않은 경우에만 변환
if aqua_train:
    print("Converting training data to Alpaca format...")
    alpaca_train = convert_to_alpaca_format(aqua_train)

if aqua_val:
    print("Converting validation data to Alpaca format...")
    alpaca_val = convert_to_alpaca_format(aqua_val)

if aqua_test:
    print("Converting test data to Alpaca format...")
    alpaca_test = convert_to_alpaca_format(aqua_test)

# 변환된 데이터 저장
def save_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved converted data to {file_path}")

if alpaca_train:
    save_data(alpaca_train, 'D:/llm_AQUA_Alpaca/alpaca_train.json')

if alpaca_val:
    save_data(alpaca_val, 'D:/llm_AQUA_Alpaca/alpaca_val.json')

if alpaca_test:
    save_data(alpaca_test, 'D:/llm_AQUA_Alpaca/alpaca_test.json')

print("Data conversion complete!")
