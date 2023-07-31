import datasets
import torch
from datasets import load_dataset, ClassLabel, Features, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorForSeq2Seq

from models import ZeroShotModels

candidate_labels = [
    "attack",
    "normal"
]
model_features = Features({'text': Value('string'), 'label': ClassLabel(names=candidate_labels)})
OUTPUT_DIR= "test_trainer"

def get_training_args():
    return TrainingArguments(output_dir=OUTPUT_DIR,
                             logging_strategy="epoch",
                             label_names=candidate_labels,
                             load_best_model_at_end=True,
                             )


def get_data_set(from_percent, to_percent, filename, seed=42):
    return load_dataset("niting3c/malicious-packet-analysis",
                        features=model_features,
                        column_names=['text', 'label'],
                        data_files={"train": filename},
                        split=datasets.ReadInstruction("train",
                                                       from_=from_percent,
                                                       to=to_percent,
                                                       unit="%",
                                                       rounding="pct1_dropremainder")
                        ).map(tokenize_function, batched=True).shuffle(seed=seed)


zero_shot = ZeroShotModels()
model_entry = zero_shot.get_models_by_suffix("llama-2-7b")[0]
tokenizer = AutoTokenizer.from_pretrained(model_entry["model_name"])


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


normal_dataset_0_70_train = get_data_set(0, 70, "data/normal.csv")

normal_dataset_70_90_test = get_data_set(70, 90, "data/normal.csv")

normal_dataset_validate = get_data_set(90, 100, "data/normal.csv")

mixed_dataset_0_70_train = get_data_set(0, 70, "data/mixed_data.csv")

mixed_dataset_70_90_test = get_data_set(70, 90, "data/mixed_data.csv")

mixed_dataset_validate = get_data_set(90, 100, "data/mixed_data.csv")

model_entry["model"] = AutoModelForSequenceClassification.from_pretrained(model_entry["model_name"], num_labels=2)
training_args = get_training_args()

data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)
trainer = Trainer(
    model=model_entry["model"],
    train_dataset=[normal_dataset_0_70_train, mixed_dataset_0_70_train],
    test_dataset=[normal_dataset_70_90_test, mixed_dataset_70_90_test],
    eval_dataset=[normal_dataset_validate, mixed_dataset_validate],
    args=get_training_args(),
    data_collator=data_collator
)
model_entry["model"].config.use_cache = False
trainer.train()
trainer.evaluate()

model_entry["model"].save_pretrained(OUTPUT_DIR)

model = torch.compile(model_entry["model"])

model.push_to_hub("niting3c/llama-2-7b-hf-zero-shot", use_auth_token=True)
tokenizer.push_to_hub("niting3c/llama-2-7b-hf-zero-shot", use_auth_token=True)