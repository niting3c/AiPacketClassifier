import transformers

model = transformers.AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b")

ZERO_SHOT = "zero-shot-classification"
ATTACK = "attack"
NORMAL = "normal"

candidate_labels = [
    ATTACK,
    NORMAL
]

label2id = {label: i for i, label in enumerate(candidate_labels)}

# Set the model's label mapping
model.config.label2id = label2id

# Convert the model to zero-shot mode
model.config.zero_shot_classification = True

model.push_to_hub("llama-2-7b-hf-zero-shot", use_auth_token=True)