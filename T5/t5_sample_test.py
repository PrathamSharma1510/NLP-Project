# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from datasets import load_dataset

# # Load the fine-tuned model and tokenizer
# model_path = './fine_tuned_model'  # Ensure this is the correct path to your saved model
# tokenizer = T5Tokenizer.from_pretrained(model_path)
# model = T5ForConditionalGeneration.from_pretrained(model_path)
# model.eval()  # Set the model to evaluation mode

# # Load the test data
# sdoh_data_path = 'Iteration__1.csv'  # Adjust as necessary
# dataset = load_dataset('csv', data_files={'data': sdoh_data_path})
# test_dataset = dataset['data'].train_test_split(test_size=0.2, seed=42)['test']

# # Prepare the test data
# max_input_length = 512
# max_target_length = 128

# def tokenize_function(examples):
#     return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_input_length)

# test_dataset = test_dataset.map(tokenize_function, batched=True)
# test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# # Function to generate predictions
# def generate_predictions(model, dataset):
#     model.eval()
#     predictions = []
#     for batch in dataset:  # Assuming dataset is a PyTorch DataLoader or similar
#         input_ids = batch['input_ids'].unsqueeze(0).to(model.device)
#         attention_mask = batch['attention_mask'].unsqueeze(0).to(model.device)

#         with torch.no_grad():
#             outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_target_length)

#         decoded_predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
#         predictions.extend(decoded_predictions)

#     return predictions


# # Generate predictions
# predictions = generate_predictions(model, test_dataset)

# # Display some predictions
# for i in range(5):  # Adjust the range as needed
#     print(f"Prediction {i+1}: {predictions[i]}")



import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
df = pd.read_csv('Iteration__1.csv')
# Load the trained model and tokenizer
model_path = './fine_tuned_model'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Function to predict the SDOH category of the input text
def predict_sdoh_category(text, model, tokenizer, max_length=512):
    # Format the input text with the prompt used during training
    prompt_text = f"Given the situation: {text}, what is the SDOH category?"

    # Encode the text input and generate the prediction
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt', max_length=max_length, truncation=True)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # create attention mask

    with torch.no_grad():  # Do not calculate gradients to speed up this process
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50)  # Adjust max_length if necessary

    # Decode the generated token ids to text
    predicted_category = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return predicted_category

# Example text input
# text_to_classify = "Patient describes being forced out of previous residence by landlord who did not renew lease agreement despite timely rent payments; currently staying at homeless shelter while searching for new housing options."


for i in range(10):
    text_to_classify = df['text'][i]


    # Get the predicted SDOH category
    predicted_category = predict_sdoh_category(text_to_classify, model, tokenizer)
    print(f"Predicted SDOH Category: {predicted_category}")

