# Setup Guide: Resume Skill Gap Analyzer

Welcome to the upgraded Resume Skill Gap Analyzer! This guide will walk you through setting up the environment, configuring API keys, and running the application.

## Prerequisites
- Python 3.8 to 3.11 recommended.
- A free Google Gemini API key (from [aistudio.google.com](https://aistudio.google.com/app/apikey)).

## Step 1: Install Dependencies
Open your terminal/command prompt in the `resume-skill-gap-analyzer-main` directory and run:

```bash
pip install -r requirements.txt
```

**Note on spaCy:** To use the advanced NLP skill extraction, download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```
*(If this fails, the app will gracefully fall back to regex-based extraction.)*

## Step 2: Configure API Keys
We've added a `.env` file in the root directory. Open it and paste your Google Gemini API key:
```env
GEMINI_API_KEY=your_actual_key_here
```
*(If you don't provide a key, AI features like the Chatbot, Career Coach, and Rewriter will automatically be disabled, and the app will continue to run).*

## Step 3: Train the Models (Optional but Recommended)
The app comes with a brand new training script `train_model_v2.py` that supports:
- Stratified 5-Fold Cross Validation.
- Training on Kaggle Data (if `Resume.csv` is present).
- Saving a `training_report.txt` with evaluation metrics.

Run it using:
```bash
python train_model_v2.py
```
*(If you do not run this, make sure `best_model.pkl` and `tfidf_vectorizer.pkl` are in your folder from the previous versions).*

## Step 4: Run the Web App
Start the Streamlit interface:
```bash
streamlit run app.py
```

## Troubleshooting
- **Missing Models?** Ensure the `pkl` files are generated via `train_model_v2.py`.
- **Missing Libraries?** Check the Streamlit sidebar inside the app—it has a checklist showing exactly which parsing libraries (`pdfplumber`, `python-docx`, etc.) are installed and active.
