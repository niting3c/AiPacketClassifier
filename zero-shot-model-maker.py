import transformers
import torch

tokenizer = transformers.LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

model = transformers.LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

ZERO_SHOT = "zero-shot-classification"
ATTACK = "attack"
NORMAL = "normal"

candidate_labels = [
    ATTACK,
    NORMAL
]

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2


label2id = {label: i for i, label in enumerate(candidate_labels)}

# Set the model's label mapping
model.config.label2id = label2id

# Convert the model to zero-shot mode
model.config.zero_shot_classification = True

model = model.eval()
model = torch.compile(model)

model.push_to_hub("niting3c/llama-2-7b-hf-zero-shot", use_auth_token=True)
