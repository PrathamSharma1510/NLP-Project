from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

# Configuration
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
max_input_length = 512
max_target_length = 128
training_batch_size = 4
sdoh_data_path = 'Iteration__1.csv'

model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load and preprocess the datasets
dataset = load_dataset('csv', data_files={'data': sdoh_data_path})
train_test_split = dataset['data'].train_test_split(test_size=0.2)
tokenized_datasets = train_test_split.map(
    lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_input_length),
    batched=True
)
tokenized_datasets = tokenized_datasets.map(
    lambda batch: {"labels": tokenizer(batch["adverse"], truncation=True, padding="max_length", max_length=max_target_length).input_ids},
    batched=True
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Instantiate the model
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results_adverse',
    num_train_epochs=3,
    per_device_train_batch_size=training_batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
)

# Define trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained('./fine_tuned_model_adverse')
tokenizer.save_pretrained('./fine_tuned_model_adverse')
