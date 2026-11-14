"""Auto-generated reproducible Hugging Face PEFT / TRL training script."""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42
torch.manual_seed(SEED)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","v_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=8,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=0.02,
    weight_decay=0.005,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    max_seq_length=256,
)
trainer.train()
model.save_pretrained("./adapter")
tokenizer.save_pretrained("./tokenizer")
