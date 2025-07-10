# Legal Document Summarizer

This project presents a Legal Document Summarizer that uses Natural Language Processing (NLP) to automatically generate summaries of legal case documents.
The summarization can be done in two ways **Whole Summary** and **Segmented Summary** based on user preference.
It is designed to assist legal professionals in quickly understanding lengthy legal texts.

---

## Project Description

The goal of this project is to provide accurate and meaningful summaries of legal documents, which are often long and complex.
Users can upload a `.txt` file or paste legal text into the application and choose between:

- **Whole Summary**: A short, coherent summary of the entire document in one concise passage.
- **Segmented Summary**: A section-wise breakdown of the document organized into legal categories such as Facts, Analysis, Judgment, Arguments, and Statute.

The project includes both **extractive** and **abstractive** summarization techniques by combining domain-specific language models with structured preprocessing and classification workflows.

---

## Tools and Technologies Used

### 1. Fine-tuned LegalBERT Models
We fine-tuned two separate LegalBERT models:

- **Extractive Model**: Trained to extract legally significant sentences from the document. This ensures key details are retained.
- **Classification Model**: Trained to categorize extracted sentences into predefined legal sections (e.g., Facts, Judgment).  
  - Both models were fine-tuned on **100 legal case files** ensuring high-quality and accurate training data.

### 2. Fine-tuned BART Model
- Used for generating **abstractive summaries**.
- Trained on **7,100 legal case documents** along with their summaries.
- Produces fluent, readable summaries that maintain legal accuracy and coherence.

### 3. Flask (Python Backend)
- Provides an API to handle user inputs, model inference, and deliver output to the frontend.
- Handles data preprocessing, including cleaning and segmentation using **spaCy**.

### 4. Web Interface (HTML/CSS/JavaScript)
- Allows users to upload `.txt` files or paste raw text.
- Users can select their preferred summarization type.
- Displays the result in-browser and provides an option to download it as a `.txt` file.

### 5. Evaluation Metrics
- We used **ROUGE** and **BLEU** scores to evaluate the quality and relevance of the generated summaries.
- These metrics assess the overlap and fluency of machine-generated summaries compared to reference summaries.

---

## How It Works

1. The user uploads or pastes a legal document.
2. The backend processes the input using **spaCy** for cleaning of case file.
3. Based on the user’s choice:
   - For **Whole Summary**, the BART model generates a short abstractive summary.
   - For **Segmented Summary**, LegalBERT extracts and classifies key sentences, which are then summarized using the BART model.
4. The summary is displayed and can be downloaded.

---


## Citation

The datasets used in this project are sourced from Shukla et al. (2022), titled Legal Case Document Summarization: Extractive and Abstractive Methods and Their Evaluation. 
```bibtex
@inproceedings{shukla2022,
  title={Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation},
  author={Shukla, Abhay and Bhattacharya, Paheli and Poddar, Soham and Mukherjee, Rajdeep and Ghosh, Kripabandhu and Goyal, Pawan and Ghosh, Saptarshi},
  booktitle={The 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing},
  year={2022}
}
