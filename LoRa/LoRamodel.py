from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
import json
from datasets import Dataset

# 모델 및 토크나이저 로드
model_name = "gpt-3"  # 필요한 모델로 변경 가능
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)

# 데이터 로드
with open('alpaca_train.json', 'r', encoding='utf-8') as f:
    alpaca_train = json.load(f)

with open('sharegpt_korean.json', 'r', encoding='utf-8') as f:
    sharegpt_train = json.load(f)

# 데이터 병합
combined_train = alpaca_train + sharegpt_train

# 데이터셋 변환
train_dataset = Dataset.from_list(combined_train)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 모델 학습
trainer.train()
