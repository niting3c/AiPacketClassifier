import gc

import datasets
import torch
from datasets import load_dataset
from peft import (
    LoraConfig
)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer

model_name = "niting3c/llama-2-7b-hf-zero-shot-prompt"
dataset_name = "niting3c/malicious-packet-analysis"
new_model = "niting3c/llama-2-7b-hf-zero-shot-prompt"
model_path = "/content/drive/MyDrive/llama-2-7b-custom-new"  # change to your preferred path
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 2
fp16 = False
bf16 = False
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 100
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.01
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 40
logging_steps = 40
max_seq_length = 1600
packing = False
device_map = {"": 0}

candidate_labels = ["attack", "normal"]

model_features = datasets.Features(
    {'text': datasets.Value('string'), 'label': datasets.ClassLabel(num_classes=2, names=candidate_labels)})

train_dataset = load_dataset("niting3c/Malicious_packets", split="train", features=model_features).shuffle(
    seed=1193).select(range(10000))


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['text'])):
        text = f"### prompt: {example['text'][i]}\n ### Response: {example['label'][i]}"
        output_texts.append(text)
    return output_texts


torch.cuda.empty_cache()

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
        bf16 = True

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    num_labels=2
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="all",
    evaluation_strategy="steps",
    hub_model_id="niting3c/llama-2-7b-hf-zero-shot-prompt",
    push_to_hub=True,
    load_best_model_at_end=True,
    eval_steps=10  # Evaluate every 20 steps
)
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=formatting_prompts_func,
    packing=packing,
)
#

torch.cuda.empty_cache()
gc.collect()

print("Starting the training")
trainer.train()
trainer.save_model("./results")

trainer.model.save_pretrained(new_model)
