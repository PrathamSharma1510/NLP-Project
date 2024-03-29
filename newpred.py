import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch

def load_model_and_tokenizer(model_path):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        return model, tokenizer
    except ImportError as e:
        print("Required libraries not found. Ensure you have SentencePiece installed.")
        print(e)
        exit()

def predict_sodh(text, model, tokenizer, device):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main(model_path, input_csv, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.to(device)

    data = pd.read_csv(input_csv)
    # Assuming 'section_text' and 'dialogue' are the columns you want to concatenate for prediction
    data['combined_text'] = data.apply(lambda row: f"{row['section_text']} {row['dialogue']}", axis=1)

    predictions = [predict_sodh(text, model, tokenizer, device) for text in data['combined_text'].tolist()]
    
    data['SDOH_predictions'] = predictions
    data.to_csv(output_csv, index=False)

if __name__ == "__main__":
    model_path = "output"  # Update this path
    input_csv = "MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv"
    output_csv = "output1/SDOH_predictions.csv"
    main(model_path, input_csv, output_csv)
