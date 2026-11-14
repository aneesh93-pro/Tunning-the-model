"""Production GPU PyTorch + PEFT + BitsAndBytes QLoRA training pipeline."""
from dataclasses import dataclass
from typing import List
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


@dataclass
class PeftJobConfig:
    model_id: str
    train_file: str
    val_file: str
    output_dir: str
    method: str = "qlora"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 512
    seed: int = 42


def run_sft_training(cfg: PeftJobConfig) -> dict:
    torch.manual_seed(cfg.seed)
    bnb_config = None
    if cfg.method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if cfg.method == "qlora":
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.target_modules or ["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    ds = load_dataset("json", data_files={"train": cfg.train_file, "validation": cfg.val_file})
    args = TrainingArguments(
        output_dir=f"{cfg.output_dir}/checkpoints",
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        seed=cfg.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        max_seq_length=cfg.max_seq_length,
    )
    result = trainer.train()
    model.save_pretrained(f"{cfg.output_dir}/adapter")
    tokenizer.save_pretrained(f"{cfg.output_dir}/tokenizer")
    return {"train_loss": result.training_loss}
