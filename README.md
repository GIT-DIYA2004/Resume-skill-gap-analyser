# Resume Skill Gap Analyzer (Pro Edition)

An advanced, AI-powered Streamlit web application designed to help job seekers perfectly tailor their resumes to their target roles. By leveraging deep learning semantic embeddings and Google's Gemini API, this tool analyzes your resume against real job descriptions to identify missing skills, generate personalized learning roadmaps, and instantly draft customized resume rewrites.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)

##  Key Features

- **Deep Learning Matching:** Uses Hugging Face `sentence-transformers` for robust semantic searching and TF-IDF vectors to accurately match your resume to the best fitting jobs.
- **AI Career Coach:** Native chat interface powered by Gemini to provide tailored career advice, interview questions, and 30-day learning roadmaps based exactly on your skill gaps.
- **Advanced Parsing:** Native support for uploading standard `.txt`, `.pdf` (`pdfplumber`), and `.docx` (`python-docx`) files.
- **Resume Quality Scorer:** Evaluates your resume on word count, action verbs, quantified metrics, and section completeness to give you a definitive overall strength score.
- **Smart Extraction:** Extracts Degrees, Institutions, Experience, and intelligently splits your resume into functional sections (Summary, Experience, Projects).
- **Live Remote Jobs:** Toggle integration with the **Remotive API** to test your resume against real, live remote developer jobs.
- **Export & Rewrite:** Download beautiful PDF / TXT analysis reports, or let the AI automatically draft a tailored version of your resume targeting your top-matched job.

##  Quick Start

For detailed setup instructions, please see the comprehensive [SETUP_GUIDE.md](SETUP_GUIDE.md).

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Environment
Create a `.env` file in the root directory (or use the one provided) and add your free Google Gemini API Key:
```env
GEMINI_API_KEY=your_actual_key_here
```

### 3. Run the App
```bash
streamlit run app.py
```

##  Model Training Architecture
The backend NLP model runs on a massive synthetic text embedding strategy. The native `train_model_v2.py` script automatically ingests the tabular `resume_dataset_200k_enhanced.csv` dataset, formats the numerical/categorical features into generated rich-text paragraphs, and trains a `LogisticRegression`/`SVC` baseline validated using 5-Fold Stratified Cross-Validation.

**To trigger a manual retraining of the core NLP vectorizer:**
```bash
python train_model_v2.py
```
