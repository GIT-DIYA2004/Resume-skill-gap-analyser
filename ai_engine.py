import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Load environment variables from the .env file automatically
load_dotenv()

def configure_gemini():
    """Initializes the Gemini LLM via LangChain from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    
    # Check if key exists and isn't your placeholder
    if api_key :
    
        try:
            # Initialize the LangChain Chat Model
            client = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0.7,
                # Note: max_output_tokens can be set here or dynamically during the call
            )
            return client, True
        except Exception as e:
            print(f"Gemini configuration failed: {e}")
            return None, False
    return None, False

def ask_gemini(client, prompt, max_tokens=1000):
    """Core wrapper for generating content using LangChain."""
    if not client:
        return "Gemini API Not Configured. Please check your API key."
    try:
        # Pass max_tokens dynamically to the invoke method
        response = client.invoke(prompt, max_output_tokens=max_tokens)
        
        # LangChain returns an AIMessage object, so we extract the .content
        return response.content
    except Exception as e:
        return f"Error connecting to Gemini: {e}"

def ai_rewrite_resume(client, resume_text, job_title, company, matched_skills):
    if not client:
        return f"Fallback Manual Rewrite for {job_title} at {company}...\nFocus on highlighting these skills: {', '.join(matched_skills)}."
    
    prompt = f"""
    Rewrite the following resume to better target the role of '{job_title}' at '{company}'.
    Highlight the following skills prominently: {matched_skills}.
    Keep it professional, concise, and structured.
    
    Original Resume:
    {resume_text[:2000]}
    """
    return ask_gemini(client, prompt, max_tokens=1500)

def ai_interview_questions(client, job_title, resume_text):
    if not client:
        return "Ask behavioral questions like 'Tell me about a time you...' and technical questions related to your field."
        
    prompt = f"""
    Given the applicant's resume summary: {resume_text[:1000]}
    Generate 5 tailored interview questions and short advice on how to answer them for the role of '{job_title}'.
    """
    return ask_gemini(client, prompt)

def ai_learning_roadmap(client, missing_skills, target_job):
    if not client:
        return "Please study the missing skills using free resources like YouTube and freeCodeCamp."
        
    prompt = f"""
    The user wants to become a '{target_job}' but is missing these skills: {missing_skills[:10]}.
    Provide a concise 30-day learning roadmap with specific free resources mentioned (YouTube, Codecademy, freeCodeCamp, Kaggle).
    """
    return ask_gemini(client, prompt)

def ai_chat(client, user_q, job_title):
    """Handles the Career Coach chatbot logic."""
    if not client:
        return "Gemini API not active. Add key to .env."
    
    # In LangChain, it's often better to pass explicit System and Human messages for chat,
    # but to keep it simple and match your previous logic, a combined prompt works perfectly.
    prompt = f"System Context: The user has a resume matching '{job_title}'. Help them with their career question: {user_q}"
    return ask_gemini(client, prompt, max_tokens=500)


# Add this to the very bottom of ai_engine.py for local testing
if __name__ == "__main__":
    print("Testing LangChain Gemini Connection...")
    client, is_connected = configure_gemini()
    
    if is_connected:
        print("✅ Connection Successful!")
        # Test a quick prompt
        response = ask_gemini(client, "Reply with 'Engine is online!'", max_tokens=10)
        print(f"🤖 AI Response: {response}")
    else:
        print("❌ Connection Failed. Check your .env file and GEMINI_API_KEY.")