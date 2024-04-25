# NLP-Healers
Here's a template for a README.md that you can use for your GitHub repository. It outlines the project structure, methodology, and results of your NLP project, which focuses on fine-tuning and utilizing various language models for extracting Social Determinants of Health (SDoH) from clinical notes.

---

# NLP Project on SDoH Label Extraction

## Project Overview

This project aims to evaluate and compare the effectiveness of various language models (including GPT-2, GPT-3.5 Turbo, GPT-4 Turbo, and Gemma 7B) in identifying Social Determinants of Health (SDoH) from electronic health records (EHR). The project involves fine-tuning a GPT-2 model on a specialized SDoH dataset, augmenting data with GPT-3.5, and performing inference using multiple state-of-the-art models.

## Directory Structure

```
NLP-Project
│
├── README.md               # Project overview and documentation
├── requirements.txt        # Python dependencies for the project
├── temp_requirements       # Temporary requirements
│
├── dataset files           # Folder containing datasets used in the project
├── T5                      # T5 model scripts and notebooks (if applicable)
├── gpt2 finetune           # Scripts and notebooks for fine-tuning the GPT-2 model
├── gpt3.5_turbo            # Scripts for making API calls to GPT-3.5 Turbo
├── gpt4-turbo              # Scripts for making API calls to GPT-4 Turbo
├── gemma                   # Scripts for making API calls to Gemma 7B model
├── augument_data.ipynb     # Notebook for augmenting MTS dataset
```

## Methodology

### Data Preparation and Augmentation
- **Dataset Files:** Contain the original and augmented datasets.
- **augument_data.ipynb:** Jupyter notebook used to augment the MTS dataset with SDoH information using GPT-3.5.

### Model Fine-Tuning and Inference
- **gpt2 finetune:** Contains notebooks and scripts for training the GPT-2 model on SDoH data.
- **gpt3.5_turbo, gpt4-turbo, gemma:** Directories containing scripts to perform inference on both the SDoH test set and the augmented MTS dataset, utilizing the respective model's API.

## Results

The fine-tuned GPT-2 model demonstrated excellent performance with the following accuracies on the SHADR test set:
- **SDoH Label Accuracy:** 85.56%
- **Adverse Label Accuracy:** 91.11%

On the augmented MTS dataset, the model performances were:
- **SDoH Label Accuracy:** 69.59%
- **Adverse Label Accuracy:** 61.8%

The results illustrate the potential of advanced NLP models in enhancing the extraction of SDoH information from clinical narratives, which is pivotal for improving healthcare outcomes and equity.


## Usage

To replicate the findings or to use the models on new data, refer to individual scripts within the respective directories for detailed instructions on running the models and performing inference.

