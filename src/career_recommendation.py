import os
import json
import pandas as pd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

def conditional_cache(func):
    """
    Decorator that applies st.cache_data if streamlit runtime is active.
    Otherwise, returns the function untouched to avoid warnings/side-effects.
    """
    if st.runtime.exists():
        return st.cache_data(func)
    return func

def _generate_content_with_fallback(prompt):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No GEMINI_API_KEY found in environment.")
        
    genai.configure(api_key=api_key.strip())
    
    # List of models to try in order of preference
    models_to_try = ['gemini-3.5-flash', 'gemini-2.0-flash', 'gemini-flash-latest']
    
    last_error = None
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            last_error = e
            continue
            
    raise last_error if last_error else RuntimeError("Failed to generate content with all available models.")

def _parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```json") or lines[0].startswith("```"):
            lines = lines[1:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)

@conditional_cache
def recommend_careers(student_profile):
    """
    Computes a match percentage and specific feedback for 7 career paths.
    Uses Gemini API if available, falling back to local heuristic logic on failure.
    """
    cgpa = student_profile.get('cgpa', 7.0)
    coding = student_profile.get('coding', 50.0)
    aptitude = student_profile.get('aptitude', 50.0)
    comm = student_profile.get('communication', 50.0)
    tech_interview = student_profile.get('technical_interview', 50.0)
    projects = student_profile.get('projects', 1)
    certifications = student_profile.get('certifications', 1)
    languages = str(student_profile.get('languages', ''))
    
    prompt = f"""
You are an expert career advisory AI.
Analyze the following student profile metrics and evaluate their compatibility across these 7 career paths:
- Software Engineer
- Data Scientist
- Machine Learning Engineer
- Data Analyst
- Business Analyst
- AI Engineer
- Full Stack Developer

Student Profile:
- CGPA: {cgpa:.2f}/10.0
- Coding Competency: {coding:.1f}/100
- Aptitude Score: {aptitude:.1f}/100
- Communication: {comm:.1f}/100
- Technical Interview: {tech_interview:.1f}/100
- Number of Projects: {projects}
- Number of Certifications: {certifications}
- Known Programming Languages / Technologies: {languages}

For each of the 7 roles, calculate a compatibility match score (0.0 to 100.0 as a float), determine a fit level ("Excellent Fit" if score >= 80, "Good Fit" if score >= 60 else "Aspirational"), and generate 3 to 4 personalized, specific bullet points explaining "Why this role?" based on their metrics and language profile. Keep explanation bullets clear, positive, and constructive.

Respond STRICTLY in JSON format matching the schema below:
[
  {{
    "role": "Role Name",
    "score": 85.5,
    "fit": "Excellent Fit",
    "reasons": [
      "Reason bullet 1 detailing why coding score or projects match.",
      "Reason bullet 2 mentioning language matches.",
      "Reason bullet 3 showing growth areas or alignment."
    ]
  }},
  ...
]

Do not include any code block syntax markers (like ```json), markdown formatting, or trailing text. Return only the raw JSON string.
"""
    try:
        response_text = _generate_content_with_fallback(prompt)
        recommendations = _parse_json_response(response_text)
        
        # Validate keys and shape of recommendations
        validated_recs = []
        for item in recommendations:
            if all(k in item for k in ('role', 'score', 'fit', 'reasons')):
                # Ensure correct types
                item['score'] = float(item['score'])
                if not isinstance(item['reasons'], list):
                    item['reasons'] = [str(item['reasons'])]
                validated_recs.append(item)
                
        if len(validated_recs) == 7:
            # Sort by score descending
            validated_recs = sorted(validated_recs, key=lambda x: x['score'], reverse=True)
            return validated_recs
        else:
            raise ValueError("Invalid number of roles returned from LLM")
            
    except Exception as e:
        # Fallback to local heuristic logic on any failure
        print(f"[RECS] Career recommendation API failed: {e}. Falling back to heuristic.")
        return _recommend_careers_fallback(student_profile)

def _recommend_careers_fallback(student_profile):
    cgpa = student_profile.get('cgpa', 7.0)
    coding = student_profile.get('coding', 50.0)
    aptitude = student_profile.get('aptitude', 50.0)
    comm = student_profile.get('communication', 50.0)
    tech_interview = student_profile.get('technical_interview', 50.0)
    projects = student_profile.get('projects', 1)
    certifications = student_profile.get('certifications', 1)
    languages = str(student_profile.get('languages', '')).lower()
    
    # Career paths scoring logic
    recommendations = []
    
    # 1. Software Engineer
    se_score = (coding * 0.4) + (tech_interview * 0.3) + (aptitude * 0.15) + (min(projects, 3) * 5)
    if "java" in languages or "c++" in languages:
        se_score += 10
    if "python" in languages or "javascript" in languages:
        se_score += 5
    se_score = min(100.0, max(30.0, se_score))
    
    se_reasons = [
        f"Your coding score of {coding:.1f}/100 shows a strong foundation for software development.",
        f"You have completed {projects} projects, showing practical implementation experience.",
        "Your programming language profile matches core industry standards for application logic."
    ]
    if "java" in languages or "c++" in languages:
        se_reasons.append("Proficiency in object-oriented programming (Java/C++) is highly preferred by core software companies.")
        
    recommendations.append({
        'role': 'Software Engineer',
        'score': se_score,
        'fit': 'Excellent Fit' if se_score >= 80 else 'Good Fit' if se_score >= 60 else 'Aspirational',
        'reasons': se_reasons
    })
    
    # 2. Data Scientist
    ds_score = (cgpa * 10.0 * 0.2) + (coding * 0.25) + (aptitude * 0.25) + (tech_interview * 0.15) + (min(projects, 3) * 5)
    if "python" in languages:
        ds_score += 10
    if "sql" in languages:
        ds_score += 5
    ds_score = min(100.0, max(30.0, ds_score))
    
    ds_reasons = [
        f"Your CGPA of {cgpa:.2f} satisfies the rigorous academic requirements for data science roles.",
        f"Your aptitude score ({aptitude:.1f}/100) indicates strong analytical and statistical reasoning skills.",
        "Python and SQL are in your profile, which are the primary tools used for data manipulation and modeling."
    ]
    if projects >= 2:
        ds_reasons.append("Your multiple projects show capability in independent problem-solving and model design.")
        
    recommendations.append({
        'role': 'Data Scientist',
        'score': ds_score,
        'fit': 'Excellent Fit' if ds_score >= 80 else 'Good Fit' if ds_score >= 60 else 'Aspirational',
        'reasons': ds_reasons
    })
    
    # 3. Machine Learning Engineer
    mle_score = (coding * 0.3) + (tech_interview * 0.3) + (cgpa * 10.0 * 0.15) + (min(projects, 3) * 8)
    if "python" in languages:
        mle_score += 8
    if "c++" in languages:
        mle_score += 5
    mle_score = min(100.0, max(30.0, mle_score))
    
    mle_reasons = [
        f"Your technical interview score of {tech_interview:.1f}/100 demonstrates solid algorithmic competency.",
        f"Your coding score ({coding:.1f}) is well-suited for writing optimized model architectures.",
        "Your project history demonstrates hands-on implementation, a core requirement for ML engineering positions."
    ]
    if "python" in languages:
        mle_reasons.append("Python proficiency allows you to easily work with standard frameworks like TensorFlow/PyTorch.")
        
    recommendations.append({
        'role': 'Machine Learning Engineer',
        'score': mle_score,
        'fit': 'Excellent Fit' if mle_score >= 80 else 'Good Fit' if mle_score >= 60 else 'Aspirational',
        'reasons': mle_reasons
    })
    
    # 4. Data Analyst
    da_score = (aptitude * 0.3) + (comm * 0.25) + (coding * 0.2) + (min(projects, 3) * 6) + (min(certifications, 3) * 3)
    if "sql" in languages:
        da_score += 10
    if "python" in languages:
        da_score += 5
    da_score = min(100.0, max(30.0, da_score))
    
    da_reasons = [
        f"Your aptitude score of {aptitude:.1f}/100 shows a strong capacity for pattern recognition and quantitative analysis.",
        f"Your communication score of {comm:.1f}/100 indicates you can effectively translate technical insights to stakeholders.",
        "Your profile lists databases / SQL, which is standard for querying business databases."
    ]
    
    recommendations.append({
        'role': 'Data Analyst',
        'score': da_score,
        'fit': 'Excellent Fit' if da_score >= 80 else 'Good Fit' if da_score >= 60 else 'Aspirational',
        'reasons': da_reasons
    })
    
    # 5. Business Analyst
    ba_score = (comm * 0.4) + (aptitude * 0.3) + (coding * 0.1) + (min(projects, 3) * 6) + (min(certifications, 3) * 4)
    if "sql" in languages:
        ba_score += 6
    ba_score = min(100.0, max(30.0, ba_score))
    
    ba_reasons = [
        f"Your communication rating of {comm:.1f}/100 is excellent for business requirements gathering and presentations.",
        f"Strong aptitude performance ({aptitude:.1f}) shows you are adept at logic modeling and process formulation.",
        "Your hybrid technical and interpersonal skill-mix fits the intermediary role between engineering and management."
    ]
    
    recommendations.append({
        'role': 'Business Analyst',
        'score': ba_score,
        'fit': 'Excellent Fit' if ba_score >= 80 else 'Good Fit' if ba_score >= 60 else 'Aspirational',
        'reasons': ba_reasons
    })
    
    # 6. AI Engineer
    ai_score = (coding * 0.35) + (tech_interview * 0.3) + (min(projects, 3) * 9)
    if "python" in languages:
        ai_score += 10
    if "go" in languages or "c++" in languages:
        ai_score += 5
    ai_score = min(100.0, max(30.0, ai_score))
    
    ai_reasons = [
        f"Highly competitive coding and interview score profile matches the requirements of AI R&D labs.",
        f"Your background includes {projects} projects, indicating exposure to implementing pipelines.",
        "Proficiency in languages like Python matches modern generative AI and API deployment frameworks."
    ]
    
    recommendations.append({
        'role': 'AI Engineer',
        'score': ai_score,
        'fit': 'Excellent Fit' if ai_score >= 80 else 'Good Fit' if ai_score >= 60 else 'Aspirational',
        'reasons': ai_reasons
    })
    
    # 7. Full Stack Developer
    fs_score = (coding * 0.4) + (tech_interview * 0.2) + (min(projects, 3) * 8)
    if "javascript" in languages:
        fs_score += 12
    if "java" in languages or "python" in languages:
        fs_score += 5
    fs_score = min(100.0, max(30.0, fs_score))
    
    fs_reasons = [
        f"Coding score of {coding:.1f}/100 is solid for full-stack system architecture development.",
        f"Your portfolio contains {projects} projects, showing experience building applications.",
    ]
    if "javascript" in languages:
        fs_reasons.append("Your familiarity with JavaScript is essential for frontend frameworks (React, Angular) and backend runtime (Node.js).")
    else:
        fs_reasons.append("Adding JavaScript to your stack will significantly boost your web development capabilities.")
        
    recommendations.append({
        'role': 'Full Stack Developer',
        'score': fs_score,
        'fit': 'Excellent Fit' if fs_score >= 80 else 'Good Fit' if fs_score >= 60 else 'Aspirational',
        'reasons': fs_reasons
    })
    
    # Sort recommendations by match percentage (descending)
    recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
    return recommendations
