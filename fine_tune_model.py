import datasets
from datasets import load_dataset
from transformers import LlamaTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, \
    DataCollatorForTokenClassification

model = AutoModelForSequenceClassification.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
tokenizer = LlamaTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")

candidate_labels = ["attack", "normal"]


# defining helper functions
def get_training_args():
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


def get_data_set(fileNames, seed=42):
    model_features = datasets.Features(
        {'text': datasets.Value('string'), 'label': datasets.ClassLabel(names=candidate_labels)})
    return load_dataset("niting3c/malicious-packet-analysis",
                        data_files=fileNames,
                        features=model_features,
                        split="train",
                        ).map(tokenize_function, batched=True).shuffle(seed=seed)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=3600)


# Prepare datasets
normal_dataset_train = get_data_set(["normal_netresc/train.csv",
                                     "network-packet-flow-header-payload/train.json"], 100)
normal_dataset_validate = get_data_set(["metasploitable-data/test.csv",
                                        "network-packet-flow-header-payload/test.json"], 100)

# load the model and tokenizer

tokenizer.add_special_tokens({'pad_token': "-100"})

# Define candidate labels and model config settings
candidate_labels = ["attack", "normal"]
model.config.architectures = ["LlamaForSequenceClassification"]
model.config.zero_shot_classification = True

# Prepare training arguments and data collator
training_args = get_training_args()
data_collator = DataCollatorForTokenClassification(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

# Train and evaluate on the normal dataset
try:
    trainer = Trainer(
        model=model,
        train_dataset=normal_dataset_train,
        eval_dataset=normal_dataset_validate,
        args=training_args,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()  # Save the model after training.

    # Evaluate and report evaluation results.
    eval_results = trainer.evaluate()
    # write the eval_results into a output folder for later use
    with open('output/eval_results.txt', 'w') as f:
        print(eval_results, file=f)
    print("Evaluation results on normal dataset:", eval_results)

except Exception as e:
    print("An error occurred during training on normal dataset:")
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
