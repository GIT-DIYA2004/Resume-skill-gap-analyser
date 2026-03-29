import os
import re
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text): 
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def skill_overlap(r, j):
    return len(set(r.split()) & set(j.split()))

def title_score(r, t):
    return len(set(r.split()) & set(t.lower().split()))

def load_kaggle_data():
    """Attempt to load Kaggle dataset and Job postings to build training pairs."""
    try:
        if not os.path.exists('Resume.csv'):
            return None
            
        print("Loading Kaggle dataset Resume.csv...")
        resumes_df = pd.read_csv('Resume.csv')
        # Structure expects Columns: ID, Resume_str, Resume_html, Category
        resume_texts = resumes_df['Resume_str'].apply(clean_text)
        categories = resumes_df['Category'].str.lower()
        
        # Load jobs
        if os.path.exists('cleaned_jobs_deduped.csv'):
            jobs_df = pd.read_csv('cleaned_jobs_deduped.csv')
        elif os.path.exists('job_postings.csv'):
            jobs_df = pd.read_csv('job_postings.csv')
        else:
            print("No job dataset found for pairs. Falling back...")
            return None
            
        jobs_df['combined'] = jobs_df.get('Job Title', '') + ' ' + jobs_df.get('description', '')
        jobs_df['combined'] = jobs_df['combined'].apply(clean_text)
        
        print(f"Building training pairs... Found {len(resumes_df)} Kaggle resumes and {len(jobs_df)} jobs.")
        
        pairs = []
        labels = []
        neg_ratio = 3
        
        # Limit the number of generated pairs to prevent massive RAM usage
        # We'll aim for about 2000-5000 instances.
        np.random.seed(42)
        target_positives = 1000
        positives_found = 0
        
        # Creating positive pairs
        for idx, resume_row in resumes_df.sample(frac=1, random_state=42).iterrows():
            if positives_found >= target_positives:
                break
                
            cat = str(resume_row['Category']).lower()
            res_text = clean_text(resume_row.get('Resume_str', ''))
            if not res_text.strip():
                continue
            
            # Find jobs matching the category
            matched_jobs = jobs_df[jobs_df['Job Title'].str.lower().str.contains(cat, na=False)]
            if not matched_jobs.empty:
                job_text = matched_jobs.sample(1)['combined'].values[0]
                job_title = matched_jobs.sample(1)['Job Title'].values[0]
                
                pairs.append({'Resume_Text': res_text, 'job_text': job_text, 'Job_Title': job_title})
                labels.append(1)
                positives_found += 1
                
                # Create negative pairs
                unmatched_jobs = jobs_df[~jobs_df['Job Title'].str.lower().str.contains(cat, na=False)]
                if not unmatched_jobs.empty:
                    neg_samples = unmatched_jobs.sample(min(neg_ratio, len(unmatched_jobs)))
                    for _, neg_row in neg_samples.iterrows():
                        pairs.append({
                            'Resume_Text': res_text, 
                            'job_text': neg_row['combined'], 
                            'Job_Title': neg_row['Job Title']
                        })
                        labels.append(0)
                        
        df = pd.DataFrame(pairs)
        df['Match_Label'] = labels
        
        print(f"Successfully generated {len(df)} training pairs from Kaggle data.")
        return df
    except Exception as e:
        print(f"Error loading Kaggle data: {e}")
        return None

def load_manual_data():
    """Load the manual training set."""
    print("Falling back to manual_test_set.csv...")
    if not os.path.exists('manual_test_set.csv'):
        raise FileNotFoundError("manual_test_set.csv not found!")
    
    # NOTE FOR USER: manual_test_set.csv only has 30 rows in the original repo.
    # This might lead to overfitting, but we will evaluate with CV anyway.
    print("WARNING: manual_test_set.csv usually has ~30 truncated rows! This is a very small dataset for CV.")
    
    df = pd.read_csv('manual_test_set.csv')
    
    # Attempt to join job descriptions if missing
    if 'job_text' not in df.columns:
        if os.path.exists('cleaned_jobs_deduped.csv'):
            jobs = pd.read_csv('cleaned_jobs_deduped.csv')
            jobs['combined'] = jobs.get('Job Title', '') + ' ' + jobs.get('description', '')
            job_lookup = dict(zip(jobs['Job Title'], jobs['combined']))
            df['job_text'] = df['Job_Title'].map(job_lookup).fillna(df['Job_Title'])
        else:
            df['job_text'] = df['Job_Title']
            
    df.dropna(subset=['job_text', 'Resume_Text'], inplace=True)
    return df

def load_200k_data():
    """Load the 200k tabular dataset and create synthetic text representations."""
    if not os.path.exists('resume_dataset_200k_enhanced.csv'):
        return None
    print("Loading 200k enhanced dataset...")
    df = pd.read_csv('resume_dataset_200k_enhanced.csv')
    
    # Balanced sampling to avoid memory overload
    pos = df[df['hired'] == 1]
    neg = df[df['hired'] == 0]
    
    pos_sample = pos.sample(min(1500, len(pos)), random_state=42)
    neg_sample = neg.sample(min(1500, len(neg)), random_state=42)
    df = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42)
    
    pairs = []
    
    for _, row in df.iterrows():
        # Synthesize resume text from numerical/categorical features
        res_text = (f"Education: {row['education_level']} from {row['university_tier']}. "
                    f"CGPA: {row['cgpa']}. Experience: {row['experience_years']} years. "
                    f"Internships: {row['internships']}, Projects: {row['projects']}. "
                    f"Programming languages: {row['programming_languages']} technologies. "
                    f"Certifications: {row['certifications']}. "
                    f"Hackathons: {row['hackathons']}, Papers: {row['research_papers']}. "
                    f"Skills Score: {row['skills_score']}. "
                    f"Age: {row['age']}. Soft Skills: {row['soft_skills_score']}.")
        
        # Synthesize job description text
        company = row['company_type']
        job_title = f"{company} Role"
        job_text = f"We are a {company} looking for a candidate with strong education, high cgpa, experience and solid projects. Must have proficient programming languages and certifications."
        
        pairs.append({
            'Resume_Text': res_text.lower(),
            'job_text': job_text.lower(),
            'Job_Title': job_title.lower(),
            'Match_Label': row['hired']
        })
        
    out_df = pd.DataFrame(pairs)
    print(f"Successfully generated {len(out_df)} training pairs from the 200k dataset.")
    return out_df

def main():
    print("Starting Model Training Pipeline v2")
    
    # 1. Load Data
    data_df = load_200k_data()
    if data_df is None or len(data_df) < 50:
        data_df = load_kaggle_data()
    if data_df is None or len(data_df) < 50:
        data_df = load_manual_data()
        
    data_df = shuffle(data_df, random_state=42).reset_index(drop=True)
    
    # 2. Vectorization
    print("Vectorizing text data...")
    combined_text = pd.concat([data_df['Resume_Text'], data_df['job_text']])
    vectorizer = TfidfVectorizer(max_features=3000)
    vectorizer.fit(combined_text)
    
    resume_vecs = vectorizer.transform(data_df['Resume_Text'])
    job_vecs = vectorizer.transform(data_df['job_text'])
    
    # 3. Compute Features
    print("Computing similarity features...")
    cos_sim = [cosine_similarity(resume_vecs[i], job_vecs[i])[0][0] for i in range(len(data_df))]
    overlap = [skill_overlap(data_df.iloc[i]['Resume_Text'], data_df.iloc[i]['job_text']) for i in range(len(data_df))]
    t_match = [title_score(data_df.iloc[i]['Resume_Text'], data_df.iloc[i]['Job_Title']) for i in range(len(data_df))]
    
    X = pd.DataFrame({
        'tfidf_sim': cos_sim,
        'skill_overlap': overlap,
        'title_match': t_match
    })
    y = data_df['Match_Label']
    
    # 4. Cross-Validation Setup
    print("Evaluating models with 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'f1', 'precision', 'recall', 'average_precision']
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
    }
    
    best_model_name = None
    best_mean_f1 = -1
    best_model_obj = None
    
    report_lines = []
    report_lines.append("Training Report (Cross-Validation)")
    report_lines.append("="*40)
    report_lines.append(f"Total pairs evaluated: {len(data_df)}")
    report_lines.append(f"Positives: {sum(y==1)} | Negatives: {sum(y==0)}")
    report_lines.append("-" * 40)
    
    # Evaluate each model
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_estimator=True)
        
        f1_mean = np.mean(cv_results['test_f1'])
        f1_std = np.std(cv_results['test_f1'])
        acc_mean = np.mean(cv_results['test_accuracy'])
        prec_mean = np.mean(cv_results['test_precision'])
        rec_mean = np.mean(cv_results['test_recall'])
        ap_mean = np.mean(cv_results['test_average_precision'])
        
        msg = (f"{name}:\n"
               f"  F1-Score:    {f1_mean:.4f} ± {f1_std:.4f}\n"
               f"  Accuracy:    {acc_mean:.4f}\n"
               f"  Precision:   {prec_mean:.4f}\n"
               f"  Recall:      {rec_mean:.4f}\n"
               f"  Avg Precision: {ap_mean:.4f}\n")
        print(msg)
        report_lines.append(msg)
        
        if f1_mean > best_mean_f1:
            best_mean_f1 = f1_mean
            best_model_name = name
            # We select the estimator from the fold with the best F1 score
            best_fold_idx = np.argmax(cv_results['test_f1'])
            best_model_obj = cv_results['estimator'][best_fold_idx]
            
    # 5. Save best model and vectorizer
    print("-" * 40)
    print(f"Best Model: {best_model_name} (Mean F1: {best_mean_f1:.4f})")
    report_lines.append("-" * 40)
    report_lines.append(f"Selected Best Model: {best_model_name} (F1: {best_mean_f1:.4f})")
    
    with open('training_report.txt', 'w') as f:
        f.write("\n".join(report_lines))
        
    joblib.dump(best_model_obj, 'best_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Saved 'best_model.pkl', 'tfidf_vectorizer.pkl', and 'training_report.txt'.")
    
if __name__ == "__main__":
    main()
