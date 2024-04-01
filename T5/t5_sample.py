# import pandas as pd
# from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
# from datasets import load_dataset, DatasetDict, Dataset

# import collections.abc
# if not hasattr(collections, 'Container'):
#     collections.Container = collections.abc.Container
    
# # Load dataset
# data_path = 'Iteration__1.csv'
# df = pd.read_csv(data_path)

# # Define SDOH categories if needed directly in labels
# # Make sure the 'label' column in your CSV contains these categories directly

# # Preprocessing function to prepare data
# def preprocess_data(row):
#     # Append '_ADVERSE' or '_NONADVERSE' based on 'adverse' column
#     label_suffix = '_ADVERSE' if row['adverse'] == 'adverse' else '_NONADVERSE'
#     return {
#         "text": f"Classify the SDOH: {row['text']}",
#         "labels": row['label'] + label_suffix
#     }

# processed_data = df.apply(preprocess_data, axis=1)
# processed_df = pd.DataFrame(processed_data.tolist())

# # Training configuration
# model_name = 'google/flan-t5-base'
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)

# tokenized_datasets = Dataset.from_pandas(processed_df).map(
#     lambda examples: tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512),
#     batched=True
# )
# tokenized_datasets = tokenized_datasets.map(
#     lambda examples: {"labels": tokenizer(examples['labels'], truncation=True, padding="max_length", max_length=128).input_ids},
#     batched=True
# )

# # Training arguments
# training_args = Seq2SeqTrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     learning_rate=2e-5,
#     weight_decay=0.01,
# )

# # Initialize Trainer
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets,
#     tokenizer=tokenizer,
# )

# # Start training
# trainer.train()

# # Save the fine-tuned model and tokenizer
# model.save_pretrained('./fine_tuned_model')
# tokenizer.save_pretrained('./fine_tuned_model')





import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset

import collections.abc
if not hasattr(collections, 'Container'):
    collections.Container = collections.abc.Container

# Load and prepare data
data_path = 'Iteration__1.csv'
df = pd.read_csv(data_path)

model_name = 'google/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure labels are in the six SDOH categories
sdoh_categories = ['TRANSPORTATION', 'HOUSING', 'RELATIONSHIP', 'PARENT', 'EMPLOYMENT', 'SUPPORT']

def prepare_data(row):
    text = row['text']
    sdoh_label = row['label']  # Using 'sdoh_label' to store the SDOH category
    # print(sdoh_label)
    # Uncomment and adjust the following if needed
    # if sdoh_label not in sdoh_categories:
    #     raise ValueError(f"Label {sdoh_label} not in SDOH categories")
    prompt = f"Given the situation: {text}, what is the SDOH category? SDOH category for this note is {sdoh_label}"
    print(prompt)
    return {"text": prompt, "labels": sdoh_label}

prepared_data = df.apply(prepare_data, axis=1)

# Ensure prepared_data is a DataFrame
if isinstance(prepared_data, pd.Series):
    prepared_data_df = pd.DataFrame(prepared_data.tolist())
else:
    prepared_data_df = prepared_data

dataset = Dataset.from_pandas(prepared_data_df)

def tokenize_function(examples):
    model_inputs = tokenizer(examples['text'], max_length=512, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['labels'], max_length=128, truncation=True, padding='max_length')  # Use 'labels' from the examples
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)


# Setup the model and training
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
