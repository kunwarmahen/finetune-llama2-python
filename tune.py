import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

#https://colab.research.google.com/drive/1k0Qn7cTRNRPkaukSq0xwz2v0Gvm8bwNj#scrollTo=xZfyT0LRgf8H
#https://www.marktechpost.com/2025/02/08/fine-tuning-of-llama-2-7b-chat-for-python-code-generation-using-qlora-sfttrainer-and-gradient-checkpointing-on-the-alpaca-14k-dataset/
# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "nikhiljatiwal/minipython-Alpaca-14k"

# Fine-tuned model name
new_model = "/home/mahen/Documents/ai/finetune/llama-2/llama-2-7b-codeAlpaca"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

# ################################################################################
# # bitsandbytes parameters
# ################################################################################

# # Activate 4-bit precision base model loading
# use_4bit = True

# # Compute dtype for 4-bit base models
# bnb_4bit_compute_dtype = "float16"

# # Quantization type (fp4 or nf4)
# bnb_4bit_quant_type = "nf4"

# # Activate nested quantization for 4-bit base models (double quantization)
# use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "/home/mahen/Documents/ai/finetune/llama-2/llama-2-7b-codeAlpaca"

# Number of training epochs
num_train_epochs = 1

# Enable fp16 training (set to True for mixed precision training)
fp16 = True

# Batch size per GPU for training
per_device_train_batch_size = 2

# Batch size per GPU for evaluation
per_device_eval_batch_size = 2

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 2

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient norm (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "adamw_torch"

# Learning rate schedule
lr_scheduler_type = "constant"

# Group sequences into batches with the same length
# Saves memory and speeds up training considerably
group_by_length = True

# Ratio of steps for a linear warmup
warmup_ratio = 0.03

# Save checkpoint every X updates steps
save_steps = 100

# Log every X updates steps
logging_steps = 10

import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)

import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache()


################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}


################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load dataset
dataset = load_dataset(dataset_name, split="train")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load base model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Prepare model for training
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

from peft import get_peft_model

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Set training parameters
training_arguments = SFTConfig(
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
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    max_seq_length=max_seq_length,
    packing=packing,
    dataset_text_field="text",
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_arguments,

)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)

# Run text generation pipeline with the fine-tuned model
prompt = "How can I write a Python program that calculates the mean, standard deviation, and coefficient of variation of a dataset from a CSV file?"
pipe = pipeline(task="text-generation", model=trainer.model, tokenizer=tokenizer, max_length=400)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# secret_value_0 = user_secrets.get_secret("HF_TOKEN")

# Empty VRAM
# del model
# del pipe
# del trainer
# del dataset
del tokenizer
import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache()

import torch

# Check the number of GPUs available
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# Check if CUDA device 1 is available
if num_gpus > 1:
    print("cuda:1 is available.")
else:
    print("cuda:1 is not available.")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Specify the device ID for your desired GPU (e.g., 0 for the first GPU, 1 for the second GPU)
device_id = 1  # Change this based on your available GPUs
device = f"cuda:{device_id}"

# Load the base model on the specified GPU
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",  # Use auto to load on the available device
)

# Load the LoRA weights
lora_model = PeftModel.from_pretrained(base_model, new_model)

# Move LoRA model to the specified GPU
lora_model.to(device)

# Merge the LoRA weights with the base model weights
model = lora_model.merge_and_unload()

# Ensure the merged model is on the correct device
model.to(device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"