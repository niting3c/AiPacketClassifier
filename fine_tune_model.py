import datasets
from datasets import load_dataset
from transformers import LlamaTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, \
    DataCollatorForTokenClassification


def get_training_args():
    candidate_labels = ["attack", "normal"]
    return TrainingArguments(
        output_dir="test_trainer",
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
    model_features = datasets.Features(
        {'text': datasets.Value('string'), 'label': datasets.ClassLabel(names=candidate_labels)})
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
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=3600)


model = AutoModelForSequenceClassification.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
tokenizer = LlamaTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
tokenizer.add_special_tokens({'pad_token': "-100"})

# Define candidate labels and model config settings
candidate_labels = ["attack", "normal"]
model.config.architectures = ["LlamaForSequenceClassification"]
model.config.zero_shot_classification = True

# Prepare datasets
normal_dataset_0_90_train = get_data_set(0, 90, "data/normal.csv")
normal_dataset_validate = get_data_set(90, 100, "data/normal.csv")
mixed_dataset_0_90_train = get_data_set(0, 90, "data/mixed_data.csv")
mixed_dataset_validate = get_data_set(90, 100, "data/mixed_data.csv")

# Prepare training arguments and data collator
training_args = get_training_args()
data_collator = DataCollatorForTokenClassification(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

# Train and evaluate on the normal dataset
try:
    trainer = Trainer(
        model=model,
        train_dataset=normal_dataset_0_90_train,
        eval_dataset=normal_dataset_validate,
        args=training_args,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()  # Save the model after training.

    # Evaluate and report evaluation results.
    eval_results = trainer.evaluate()
    print("Evaluation results on normal dataset:", eval_results)

except Exception as e:
    print("An error occurred during training on normal dataset:")
    print(str(e))
    exit(1)

# Train and evaluate on the mixed dataset
try:
    trainer = Trainer(
        model=model,
        train_dataset=mixed_dataset_0_90_train,
        eval_dataset=mixed_dataset_validate,
        args=training_args,
        data_collator=data_collator,
    )

    trainer.train()
    eval_results_mixed = trainer.evaluate()
    print("Mixed data evaluation results:", eval_results_mixed)

except Exception as e:
    print("An error occurred during training on mixed dataset:")
    print(str(e))
    exit(1)

# Save the model and tokenizer to Hugging Face Hub.
model.save_pretrained("test_trainer")
tokenizer.save_pretrained("test_trainer")

# Push to Hugging Face Hub.
try:
    model.push_to_hub("niting3c/llama-2-7b-hf-zero-shot", use_auth_token=True)
    tokenizer.push_to_hub("niting3c/llama-2-7b-hf-zero-shot", use_auth_token=True)
    print("Model and tokenizer pushed to the Hugging Face Hub.")
except Exception as e:
    print("An error occurred while pushing to the Hugging Face Hub:")
    print(str(e))
    exit(1)
