import datasets
import torch
from datasets import load_dataset, ClassLabel, Features, Value
from transformers import LlamaTokenizer, TrainingArguments, Trainer, \
    LlamaForSequenceClassification, DataCollatorForTokenClassification,AutoModelForSequenceClassification

from models import ZeroShotModels
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1

candidate_labels = [
    "attack",
    "normal"
]
model_features = Features({'text': Value('string'), 'label': ClassLabel(names=candidate_labels)})
OUTPUT_DIR = "test_trainer"

def get_training_args():
    return TrainingArguments(output_dir=OUTPUT_DIR,
                             logging_strategy="epoch",
                             label_names=candidate_labels,
                             load_best_model_at_end=True,
                             evaluation_strategy="steps",
                             eval_steps=50,
                             warmup_steps=100,
                             save_steps=50,
                             num_train_epochs=10,
                             do_train=True,
                             do_eval=True,
                             )


def get_data_set(from_percent, to_percent, filename, seed=42):
    return load_dataset("niting3c/malicious-packet-analysis",
                        features=model_features,
                        data_files=filename,
                        split=datasets.ReadInstruction("train",
                                                       from_=from_percent,
                                                       to=to_percent,
                                                       unit="%",
                                                       rounding="pct1_dropremainder")
                        ).map(tokenize_function, batched=True).shuffle(seed=seed)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=False,
                       max_length=3600)


normal_dataset_0_90_train = get_data_set(0, 90, "data/normal.csv")

normal_dataset_validate = get_data_set(90, 100, "data/normal.csv")

mixed_dataset_0_90_train = get_data_set(0, 90, "data/mixed_data.csv")

mixed_dataset_validate = get_data_set(90, 100, "data/mixed_data.csv")

training_args = get_training_args()

data_collator = DataCollatorForTokenClassification(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

model.config.use_cache = False
model.config.architectures = ["LlamaForSequenceClassification", ""]
model.config.zero_shot_classification = True

trainer = Trainer(
    model=model,
    train_dataset=normal_dataset_0_90_train,
    eval_dataset=normal_dataset_validate,
    args=get_training_args(),
    data_collator=data_collator,
)

trainer.train()
trainer.evaluate()

trainer = Trainer(
    model=model,
    train_dataset=mixed_dataset_0_90_train,
    eval_dataset=mixed_dataset_validate,
    args=get_training_args(),
    data_collator=data_collator,
)

model.save_pretrained(OUTPUT_DIR)
model = torch.compile(model)
model.push_to_hub("niting3c/llama-2-7b-hf-zero-shot", use_auth_token=True)

tokenizer.save_pretrained(OUTPUT_DIR)
tokenizer.push_to_hub("niting3c/llama-2-7b-hf-zero-shot", use_auth_token=True)
