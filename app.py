import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import string
import os
import sys
import pickle
import joblib

# Initialize python-dotenv for API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ==========================================
# IMPORT NEW AI ENGINE
# ==========================================
from ai_engine import (
    configure_gemini, ai_rewrite_resume, ai_interview_questions, 
    ai_learning_roadmap, ai_chat
)

# Initialize Gemini Client globally
GEMINI_CLIENT, GEMINI_AVAILABLE = configure_gemini()

# ==========================================
# OPTIONAL DEPENDENCIES (Graceful Fallbacks)
# ==========================================
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import spacy
    try:
        nlp_spacy = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ==========================================
# CACHED SYSTEM SETUP
# ==========================================
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

download_nltk_data()

def init_session_state():
    """Initialise all session state variables safely to avoid race conditions"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'use_remotive' not in st.session_state:
        st.session_state.use_remotive = False

@st.cache_resource
def load_semantic_model():
    """Load sentence transformers model if available"""
    if ST_AVAILABLE:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model
        except Exception as e:
            return None
    return None

@st.cache_resource
def load_models_and_data():
    """Load the trained model, vectorizer, and job dataset"""
    try:
        model = None
        vectorizer = None
        
        try:
            model = joblib.load('best_model.pkl')
        except:
            for protocol in [None, 4, 3, 2]:
                try:
                    with open('best_model.pkl', 'rb') as f:
                        model = pickle.load(f) if protocol is None else pickle.load(f, encoding='latin1')
                    break
                except Exception:
                    continue
        
        try:
            vectorizer = joblib.load('tfidf_vectorizer.pkl')
        except:
            for protocol in [None, 4, 3, 2]:
                try:
                    with open('tfidf_vectorizer.pkl', 'rb') as f:
                        vectorizer = pickle.load(f) if protocol is None else pickle.load(f, encoding='latin1')
                    break
                except Exception:
                    continue
        
        # Load the job dataset
        jobs_df = pd.read_csv('cleaned_jobs_deduped.csv')
        
        if 'description' not in jobs_df.columns and 'job_description' in jobs_df.columns:
            jobs_df['description'] = jobs_df['job_description']
            
        jobs_df['combined'] = jobs_df.get('Job Title', '').fillna('') + ' ' + jobs_df.get('description', '').fillna('')
        jobs_df['skills'] = jobs_df.get('skills', jobs_df.get('description', '')).fillna('')
        
        if model is None or vectorizer is None:
            st.error("❌ Failed to load required model files. Switching to demo mode.")
            return None, None, pd.DataFrame()
            
        return model, vectorizer, jobs_df
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing crucial file. Run train_model_v2.py first.")
    except Exception as e:
        raise Exception(f"Error loading files: {str(e)}")

# ==========================================
# PARSERS & TEXT EXTRACTION
# ==========================================
def parse_resume_file(uploaded_file):
    """Parse resume extracting text based on file format."""
    filename = uploaded_file.name.lower()
    text = ""
    
    if filename.endswith(".pdf"):
        if PDF_AVAILABLE:
            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
        else:
            st.warning("`pdfplumber` is not installed! Cannot read PDF files.")
    
    elif filename.endswith(".docx"):
        if DOCX_AVAILABLE:
            try:
                doc = docx.Document(uploaded_file)
                text = "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                st.error(f"Error reading DOCX: {e}")
        else:
            st.warning("`python-docx` is not installed! Cannot read DOCX files.")
            
    else:
        try:
            text = str(uploaded_file.read(), "utf-8")
        except:
            text = str(uploaded_file.read(), "latin-1")
            
    return text

def preprocess_text(text):
    if not text: return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(processed_words)

def extract_skills(text):
    """Extract skills using regex and optionally spaCy"""
    skill_patterns = [
        r'\b(?:python|java|javascript|c\+\+|sql|html|css|react|angular|vue|node\.js|django|flask|tensorflow|pytorch|pandas|numpy|scikit-learn|docker|kubernetes|aws|azure|gcp|git|github|linux|windows|mysql|postgresql|mongodb|redis|elasticsearch|spark|hadoop|tableau|powerbi|excel|r|scala|go|rust|swift|kotlin|php|ruby|perl|bash|shell|ci/cd|devops|agile|scrum|jira|confluence|slack|trello|photoshop|illustrator|figma|sketch|autocad|solidworks|matlab|sas|spss|salesforce|sap|oracle|microsoft office|word|powerpoint|outlook|teams|zoom|slack)\b',
        r'\b(?:machine learning|deep learning|artificial intelligence|data science|data analysis|web development|mobile development|frontend|backend|full stack|software engineering|quality assurance|project management|product management|business analysis|digital marketing|social media|content creation|graphic design|ui/ux|user experience|database administration|network administration|cybersecurity|information security|cloud computing|big data|data mining|statistical analysis|financial modeling|risk management|supply chain|logistics|human resources|customer service|sales|marketing|accounting|finance)\b'
    ]
    
    skills = set()
    text_lower = text.lower()
    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        skills.update(matches)
        
    if SPACY_AVAILABLE:
        doc = nlp_spacy(text_lower)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            if len(chunk_text.split()) <= 3 and len(chunk_text) > 3:
                pass 
                
    return list(skills)

def extract_education(text):
    """Extract education details from resume"""
    ed_info = {
        'degrees': [],
        'institutions': [],
        'years': [],
        'seniority': 'Fresher'
    }
    
    degree_pattern = r'\b(?:B\.?Tech|M\.?Tech|B\.?Sc|M\.?Sc|B\.?A\.|M\.?A\.|MBA|Ph\.?D\.|Bachelor|Master|Doctorate)\b'
    ed_info['degrees'] = list(set(re.findall(degree_pattern, text, re.IGNORECASE)))
    
    inst_pattern = r'\b(?:[A-Z][a-z]+ )+(?:University|Institute|College|Academy|IIT|NIT)\b'
    ed_info['institutions'] = list(set(re.findall(inst_pattern, text)))
    
    year_pattern = r'\b(?:19|20)\d{2}\b'
    ed_info['years'] = list(set(re.findall(year_pattern, text)))
    
    exp_pattern = r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience'
    matches = re.findall(exp_pattern, text, re.IGNORECASE)
    if matches:
        try:
            max_years = max([int(m) for m in matches])
            if max_years < 2:
                ed_info['seniority'] = 'Junior'
            elif max_years <= 5:
                ed_info['seniority'] = 'Mid-level'
            else:
                ed_info['seniority'] = 'Senior'
        except:
            pass
            
    return ed_info

def parse_resume_sections(text):
    """Split resume into named sections"""
    sections = {
        'summary': '', 'experience': '', 'education': '', 'skills': '',
        'projects': '', 'certifications': '', 'other': ''
    }
    
    headers = {
        'experience': ['experience', 'work history', 'employment', 'career'],
        'education': ['education', 'academic', 'qualifications'],
        'skills': ['skills', 'technologies', 'core competencies', 'technical skills'],
        'projects': ['projects', 'personal projects', 'portfolio'],
        'certifications': ['certifications', 'certificates', 'licenses'],
        'summary': ['summary', 'objective', 'profile', 'about me']
    }
    
    current_section = 'summary'  
    lines = text.split('\n')
    
    for line in lines:
        line_clean = line.strip().lower()
        if not line_clean:
            continue
            
        is_header = False
        if len(line_clean) < 30: 
            for section, aliases in headers.items():
                if any(alias in line_clean for alias in aliases):
                    current_section = section
                    is_header = True
                    break
        
        if not is_header:
            sections[current_section] += line + '\n'
            
    return sections

def get_section_stats(sections):
    stats = {}
    for k, v in sections.items():
        stats[k] = len(v.split())
    return stats

def score_resume(text, sections):
    """Resume Quality Scorer evaluating overall strength"""
    word_count = len(text.split())
    length_score = min(word_count / 400.0, 1.0) if word_count > 100 else 0.2
    
    skills = extract_skills(text)
    skills_score = min(len(skills) / 15.0, 1.0)
    
    action_verbs = ['led', 'built', 'improved', 'managed', 'created', 'developed', 'designed', 'increased', 'reduced', 'optimized', 'delivered']
    verb_count = sum([1 for v in action_verbs if v in text.lower()])
    verb_score = min(verb_count / 5.0, 1.0)
    
    quantified = re.findall(r'\b\d{1,3}(?:%|k|m|b)\b', text.lower())
    quant_score = min(len(quantified) / 4.0, 1.0)
    
    filled_sections = sum(1 for v in sections.values() if len(v.strip()) > 20)
    section_score = min(filled_sections / 4.0, 1.0) 
    
    overall = np.mean([length_score, skills_score, verb_score, quant_score, section_score])
    
    return {
        'length': length_score, 'skills': skills_score, 'verbs': verb_score,
        'quantified': quant_score, 'sections': section_score, 'overall': overall,
        'word_count': word_count, 'verb_matches': action_verbs, 'quant_matches': quantified
    }

# ==========================================
# API INTEGRATIONS (Remotive)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_live_jobs(category="", limit=50):
    """Fetch live jobs from Remotive API"""
    if not REQUESTS_AVAILABLE: return pd.DataFrame()
    try:
        url = "https://remotive.com/api/remote-jobs"
        if category:
            url += f"?category={category}"
        url += f"&limit={limit}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            jobs = data.get('jobs', [])
            if not jobs: return pd.DataFrame()
            
            df = pd.DataFrame(jobs)
            df['Job Title'] = df['title']
            df['company'] = df['company_name']
            
            def clean_html(raw_html):
                cleanr = re.compile('<.*?>')
                cleantext = re.sub(cleanr, '', str(raw_html))
                return cleantext
                
            df['description'] = df['description'].apply(clean_html)
            df['combined'] = df['Job Title'] + ' ' + df['description']
            df['skills'] = df['description'] 
            return df
    except Exception as e:
        st.warning(f"Failed to fetch live API jobs: {e}")
    return pd.DataFrame()

# ==========================================
# ML & LOGIC FUNCTIONS
# ==========================================
def get_learning_resources(skill):
    """Expanded learning resources fallback dictionary"""
    resources = {
        "power bi": [{"title": "Power BI Full Course", "url": "https://www.youtube.com/watch?v=AGrl-H87pRU", "type": "Free YouTube"}],
        "sql": [{"title": "SQL Tutorial (W3Schools)", "url": "https://www.w3schools.com/sql/", "type": "Free"}],
        "python": [{"title": "Python for Beginners (freeCodeCamp)", "url": "https://www.youtube.com/watch?v=rfscVS0vtbw", "type": "Free"}],
        "machine learning": [{"title": "Machine Learning by Andrew Ng", "url": "https://www.coursera.org/learn/machine-learning", "type": "Free Audit"}],
        "docker": [{"title": "Docker Tutorial for Beginners", "url": "https://www.youtube.com/watch?v=3c-iBn73dDE", "type": "Free YouTube"}],
        "react": [{"title": "React Course - Beginner's Tutorial", "url": "https://www.youtube.com/watch?v=bMknfKXIFA8", "type": "Free YouTube"}],
        "aws": [{"title": "AWS Certified Cloud Practitioner Training", "url": "https://www.youtube.com/watch?v=3hLmDS179YE", "type": "Free YouTube"}],
        "kubernetes": [{"title": "Kubernetes Course - Full Tutorial", "url": "https://www.youtube.com/watch?v=X48VuDVv0do", "type": "Free YouTube"}],
        "data analysis": [{"title": "Data Analysis with Python", "url": "https://www.freecodecamp.org/learn/data-analysis-with-python/", "type": "Free"}],
        "deep learning": [{"title": "Deep Learning Crash Course", "url": "https://www.youtube.com/watch?v=vyKcZE2RkrI", "type": "Free YouTube"}],
    }
    return resources.get(skill.lower(), [
        {"title": f"Search {skill} on freeCodeCamp", "url": f"https://www.freecodecamp.org/news/search/?query={skill}", "type": "Free"},
        {"title": f"Search {skill} on YouTube", "url": f"https://www.youtube.com/results?search_query={skill}+tutorial", "type": "Free"}
    ])

def compute_features(resume_text, jobs_df, vectorizer, semantic_model=None):
    """Compute features using Semantic Embeddings if available, else TF-IDF."""
    if semantic_model is not None:
        resume_vec = semantic_model.encode([resume_text])
        job_vecs = semantic_model.encode(jobs_df['combined'].tolist())
    else:
        resume_vec = vectorizer.transform([resume_text])
        job_vecs = vectorizer.transform(jobs_df['combined'])

    tfidf_sim = cosine_similarity(resume_vec, job_vecs).flatten()

    resume_tokens = set(resume_text.lower().split())
    skill_overlap = []
    title_match = []
    
    for _, row in jobs_df.iterrows():
        job_skills = set(str(row.get('skills', row.get('description', ''))).lower().split())
        job_title = set(str(row.get('Job Title', '')).lower().split())
        skill_overlap.append(len(resume_tokens & job_skills))
        title_match.append(len(resume_tokens & job_title))

    features = pd.DataFrame({
        'tfidf_sim': tfidf_sim,
        'skill_overlap': skill_overlap,
        'title_match': title_match
    })
    return features, tfidf_sim

def apply_job_filters(jobs_df, seniority, job_type, salary_keyword, title_keyword):
    filtered_df = jobs_df.copy()
    
    if title_keyword:
        filtered_df = filtered_df[filtered_df['Job Title'].str.contains(title_keyword, case=False, na=False)]
        
    if seniority != "All":
        filtered_df = filtered_df[filtered_df['combined'].str.contains(seniority, case=False, na=False)]
        
    if job_type != "All":
        filtered_df = filtered_df[filtered_df['combined'].str.contains(job_type.replace('_',' '), case=False, na=False)]
        
    if salary_keyword:
        filtered_df = filtered_df[filtered_df['combined'].str.contains(salary_keyword, case=False, na=False)]
        
    if filtered_df.empty:
        st.warning("Filters were too restrictive. Showing all jobs instead.")
        return jobs_df
    return filtered_df

def analyze_resume(resume_text, model, vectorizer, semantic_model, jobs_df, top_n=10):
    processed_resume = preprocess_text(resume_text)
    
    features, similarities = compute_features(processed_resume, jobs_df, vectorizer, semantic_model)
    
    if model is not None:
        compatibility_scores = model.predict_proba(features)[:, 1]
    else:
        compatibility_scores = similarities 

    jobs_df = jobs_df.copy()
    jobs_df['compatibility_score'] = compatibility_scores
    jobs_df['similarity_score'] = similarities

    top_jobs = jobs_df.sort_values('compatibility_score', ascending=False).head(top_n)

    resume_skills = extract_skills(resume_text)
    all_job_skills = set()
    for _, job in top_jobs.iterrows():
        skills = extract_skills(job.get('skills', job.get('description','')))
        all_job_skills.update(skills)

    resume_skills_set = set([s.lower() for s in resume_skills])
    job_skills_set = set([s.lower() for s in all_job_skills])

    matched_skills = list(resume_skills_set.intersection(job_skills_set))
    missing_skills = list(job_skills_set - resume_skills_set)

    return {
        'top_jobs': top_jobs,
        'resume_skills': resume_skills,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'processed_resume': processed_resume,
        'sections': parse_resume_sections(resume_text),
        'education': extract_education(resume_text)
    }

def create_demo_results(resume_text):
    """Fallback if local data fails completely"""
    resume_skills = extract_skills(resume_text)
    demo_jobs = pd.DataFrame({
        'Job Title': ['Senior Data Scientist', 'AI Engineer', 'Python Dev'],
        'company': ['TechCorp', 'AI Labs', 'DevStudio'],
        'compatibility_score': [0.92, 0.85, 0.70],
        'similarity_score': [0.88, 0.82, 0.65],
        'description': ['Data scientist role...', 'AI dev...', 'Backend dev...']
    })
    
    matched_skills = ['python'] if 'python' in [s.lower() for s in resume_skills] else []
    missing_skills = ['docker', 'kubernetes', 'aws']
    
    return {
        'top_jobs': demo_jobs,
        'resume_skills': resume_skills,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'processed_resume': preprocess_text(resume_text),
        'sections': parse_resume_sections(resume_text),
        'education': extract_education(resume_text)
    }

def create_match_report(results, resume_text):
    report_content = f"JOB MATCHING ANALYSIS REPORT\n============================\n\n"
    report_content += f"Total Resume Skills Identified: {len(results['resume_skills'])}\n"
    report_content += f"Matched Skills: {len(results['matched_skills'])}\n"
    report_content += f"Missing Skills: {len(results['missing_skills'])}\n\nTOP JOB MATCHES:\n"
    
    for idx, (_, job) in enumerate(results['top_jobs'].head(10).iterrows(), 1):
        title = job.get('Job Title', 'N/A')
        company = job.get('company', 'N/A')
        compatibility = job.get('compatibility_score', 0)
        similarity = job.get('similarity_score', 0)
        report_content += f"{idx}. {title} at {company} (Comp: {compatibility:.3f}, Sim: {similarity:.3f})\n"
        
    return report_content

def create_pdf_report(results, resume_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph("Job Matching Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Top Matches", styles['Heading2']))
    
    for idx, (_, job) in enumerate(results['top_jobs'].head(5).iterrows(), 1):
        t = f"{idx}. {job.get('Job Title', 'N/A')} at {job.get('company', 'N/A')} (Score: {job.get('compatibility_score',0):.3f})"
        story.append(Paragraph(t, styles['Normal']))
        
    doc.build(story)
    buffer.seek(0)
    return buffer

# ==========================================
# MAIN UI
# ==========================================
def main():
    st.set_page_config(page_title="Resume Skill Gap Analyzer - Pro", page_icon="⚙️", layout="wide")
    
    # Init safely
    init_session_state()
    
    st.title("⚙️ Resume Skill Gap Analyzer (Pro Edition)")
    
    # Setup Models
    with st.spinner("Loading Deep Learning Semantic Models (This may take a minute on the first run as it downloads)..."):
        semantic_model = load_semantic_model()
    
    try:
        with st.spinner("Loading Job Datasets and Base Models..."):
            model, vectorizer, csv_jobs_df = load_models_and_data()
        if model is None:
            st.session_state.demo_mode = True
    except Exception as e:
        st.session_state.demo_mode = True
        st.error(f"Error loading models: {e}")
        
    # Sidebar: API & Status
    with st.sidebar:
        st.header("System Status")
        st.markdown(f"- **PDF Parsing:** {'✅' if PDF_AVAILABLE else '⚠️'}")
        st.markdown(f"- **DOCX Parsing:** {'✅' if DOCX_AVAILABLE else '⚠️'}")
        st.markdown(f"- **NLP (spaCy):** {'✅' if SPACY_AVAILABLE else '⚠️'}")
        st.markdown(f"- **Embeddings:** {'✅' if ST_AVAILABLE else '⚠️'}")
        st.markdown(f"- **Gemini AI:** {'✅' if GEMINI_AVAILABLE else '⚠️'}")
        
        st.divider()
        st.header("Data Source")
        st.session_state.use_remotive = st.toggle("Live Remote Jobs (Remotive API)", value=st.session_state.use_remotive)
        st.caption("Toggle between local CSV models and real-time live remote job listings.")
        
        top_n = st.slider("Jobs to analyze", 5, 50, 10)
        
        st.divider()
        st.header("🔽 Filter Jobs")
        f_seniority = st.selectbox("Seniority Level", ["All", "Fresher", "Junior", "Mid-level", "Senior"])
        f_type = st.selectbox("Job Type", ["All", "full_time", "contract", "part_time", "freelance"])
        f_salary = st.text_input("Salary Contains", placeholder="e.g. 100k")
        f_title = st.text_input("Title Contains", placeholder="e.g. Python")

    col1, col2 = st.columns([2, 1])
    resume_text = ""
    with col1:
        st.header("📄 Upload Resume")
        uploaded_file = st.file_uploader("Upload your resume (.txt, .pdf, .docx)", type=['txt', 'pdf', 'docx'])
        if uploaded_file:
            resume_text = parse_resume_file(uploaded_file)
            
        if not resume_text:
            resume_text = st.text_area("Or Paste Text Here:", height=200)
            
    with col2:
        st.header("📈 Quick Stats")
        if resume_text:
            st.metric("Word Count", len(resume_text.split()))
            skills_preview = extract_skills(resume_text)
            st.metric("Skills Detected", len(skills_preview))

    # Trigger Analysis
    if st.button("🔍 Analyze Resume", type="primary", use_container_width=True, disabled=not resume_text.strip()):
        with st.spinner("Analyzing..."):
            if st.session_state.demo_mode:
                st.session_state.analysis_results = create_demo_results(resume_text)
            else:
                if st.session_state.use_remotive:
                    jobs_df = fetch_live_jobs(limit=100)
                    if jobs_df.empty:
                        st.warning("API Failed. Falling back to local CSV.")
                        jobs_df = csv_jobs_df
                else:
                    jobs_df = csv_jobs_df
                    
                orig_len = len(jobs_df)
                
                jobs_df = apply_job_filters(jobs_df, f_seniority, f_type, f_salary, f_title)
                st.toast(f"Filters active: {len(jobs_df)} of {orig_len} jobs matched criteria.")
                
                res = analyze_resume(resume_text, model, vectorizer, semantic_model, jobs_df, top_n)
                st.session_state.analysis_results = res
                
    # Display Results
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.divider()
        st.header("📊 Comprehensive Findings")
        
        tabs = st.tabs([
            "🎯 Top Jobs", "🔧 Skills Analysis", "📋 Resume Sections", 
            "🎓 Education", "📊 Quality Score", "💬 AI Career Coach", 
            "🎯 Interview Prep", "📄 Export Reports"
        ])
        
        with tabs[0]: 
            st.subheader(f"Top {len(results['top_jobs'])} Matches")
            if not ST_AVAILABLE:
                st.info("⚠️ Falling back to TF-IDF. Install `sentence-transformers` for better semantic matching.")
                
            for idx, (_, job) in enumerate(results['top_jobs'].iterrows(), 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{idx}. {job.get('Job Title', 'N/A')}** @ *{job.get('company', 'Unknown')}*")
                with col2:
                    c = job.get('compatibility_score', 0)
                    st.progress(c, text=f"{c:.1%}")
                with st.expander("Details"):
                    st.write(job.get('description', 'No description...'))
                    if 'url' in job.index:
                        st.markdown(f"[Apply Here]({job['url']})")
                    
        with tabs[1]: 
            st.subheader("Skill Gap Insights")
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ Matched ({len(results['matched_skills'])})")
                st.write(", ".join(results['matched_skills']))
            with col2:
                st.error(f"❌ Missing ({len(results['missing_skills'])})")
                st.write(", ".join(results['missing_skills'][:15]))
                
        with tabs[2]: 
            st.subheader("Section Detection")
            stats = get_section_stats(results['sections'])
            for sec, words in stats.items():
                if words > 10:
                    st.success(f"{sec.title()}: {words} words")
                else:
                    st.warning(f"Missing {sec.title()} section. Consider adding it.")
            
            sel_sec = st.selectbox("View Section Content", list(results['sections'].keys()))
            st.text_area("Content", results['sections'][sel_sec], height=150)
            
        with tabs[3]: 
            st.subheader("Extracted Education Info")
            ed = results['education']
            st.write(f"**Degrees:** {', '.join(ed['degrees']) if ed['degrees'] else 'None Found'}")
            st.write(f"**Institutions:** {', '.join(ed['institutions']) if ed['institutions'] else 'None Found'}")
            st.write(f"**Years:** {', '.join(ed['years']) if ed['years'] else 'None Found'}")
            st.write(f"**Estimated Seniority Extracted:** {ed['seniority']}")
            
        with tabs[4]: 
            st.subheader("Resume Quality Scorer")
            scores = score_resume(resume_text, results['sections'])
            
            st.progress(scores['overall'], text=f"Overall Quality Score: {scores['overall']:.1%}")
            st.metric("Word Count", scores['word_count'], "Ideal is ~400+")
            st.metric("Action Verbs Detected", len(scores['verb_matches']))
            st.metric("Metrics/Quantified Data Points", len(scores['quant_matches']))
            
            if scores['overall'] < 0.7:
                st.warning("Tips to improve: Add more quantified results (e.g., 'improved by 20%') and strengthen your action verbs.")
                
        with tabs[5]: 
            st.subheader("💬 AI Career Coach")
            if not GEMINI_AVAILABLE:
                st.error("Gemini API not active. Check your .env file.")
            else:
                if st.button("Clear Chat History"):
                    st.session_state.chat_history = []
                    
                for msg in st.session_state.chat_history:
                    st.chat_message(msg['role']).write(msg['content'])
                    
                if user_q := st.chat_input("Ask for resume/career advice..."):
                    st.session_state.chat_history.append({"role": "user", "content": user_q})
                    st.chat_message("user").write(user_q)
                    
                    try:
                        top_job = results['top_jobs'].iloc[0].get('Job Title', 'Unknown Role')
                        # USE THE CLIENT
                        resp = ai_chat(GEMINI_CLIENT, user_q, top_job)
                    except Exception as e:
                        resp = f"Error: {e}"
                        
                    st.session_state.chat_history.append({"role": "assistant", "content": resp})
                    st.chat_message("assistant").write(resp)

        with tabs[6]: 
            st.subheader("Prep for your Top Match")
            top_job = results['top_jobs'].iloc[0].get('Job Title', 'N/A')
            
            if st.button("Generate Interview Questions (AI)"):
                with st.spinner("Generating..."):
                    # USE THE CLIENT
                    st.write(ai_interview_questions(GEMINI_CLIENT, top_job, resume_text))
                    
            if st.button("Generate Learning Roadmap (AI)"):
                with st.spinner("Generating..."):
                    # USE THE CLIENT
                    st.write(ai_learning_roadmap(GEMINI_CLIENT, results['missing_skills'], top_job))
                    
            st.divider()
            st.subheader("Free Learning Resources")
            for ms in results['missing_skills'][:3]:
                st.markdown(f"**{ms.upper()}**")
                for r in get_learning_resources(ms):
                    st.markdown(f"- [{r['title']}]({r['url']}) ({r['type']})")

        with tabs[7]: 
            st.subheader("Downloads & Rewrites")
            txt_report = create_match_report(results, resume_text)
            st.download_button("Download Text Report", txt_report, "report.txt")
            
            if st.button("Generate AI Tailored Resume (For Top Job)"):
                with st.spinner("Rewriting..."):
                    top_job = results['top_jobs'].iloc[0]
                    # USE THE CLIENT
                    rewritten = ai_rewrite_resume(GEMINI_CLIENT, resume_text, top_job.get('Job Title'), top_job.get('company'), results['matched_skills'])
                    st.text_area("Your Tailored Draft", rewritten, height=400)
                    st.download_button("Download Draft", rewritten, "tailored_resume.txt")

    with st.expander("🔧 Troubleshooting & Sys Info"):
        st.code(f"Python Version: {sys.version}\nPandas: {pd.__version__}")

if __name__ == "__main__":
    main()