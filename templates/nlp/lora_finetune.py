"""LoRA fine-tune a transformer. The H2O LLM Science Exam recipe, distilled.

Usage:
    python templates/nlp/lora_finetune.py
TODO: MODEL_NAME, dataset loading, and label format for your competition.
"""
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

# --- TODO: your knobs ---
MODEL_NAME = "microsoft/deberta-v3-base"  # or a 7B LLM for the full H2O recipe
NUM_LABELS = 2
MAX_LEN = 512
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05


def load_data(tokenizer):
    # TODO: replace with your competition data -> HF Dataset with
    # "input_ids", "attention_mask", "labels".
    texts = ["example text one", "example text two"]  # placeholder
    labels = [0, 1]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN)
    enc["labels"] = labels
    return Dataset.from_dict(enc)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    # LoRA: freeze the base model, train only low-rank adapters.
    # Full fine-tunes are for people with someone else's GPU budget.
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        task_type="SEQ_CLS",
        target_modules=["query", "key", "value"],  # TODO: match your architecture
    ))
    model.print_trainable_parameters()

    ds = load_data(tokenizer)
    args = TrainingArguments(
        output_dir="runs/nlp_lora",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
    )
    Trainer(model=model, args=args, train_dataset=ds).train()
    model.save_pretrained("runs/nlp_lora/adapters")
    print("[done] LoRA adapters -> runs/nlp_lora/adapters")


if __name__ == "__main__":
    main()
