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
        
    genai.configure(api_key=api_key.strip(), transport='rest')
    
    # List of models to try in order of preference
    models_to_try = ['gemini-3.5-flash', 'gemini-2.0-flash', 'gemini-flash-latest']
    
    last_error = None
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, request_options={"timeout": 5.0})
            return response.text.strip()
        except Exception as e:
            last_error = e
            # If it's a network timeout, connection error, or DNS issue, abort early to avoid long hangs
            err_msg = str(e).lower()
            if any(term in err_msg for term in ["timeout", "timed out", "connect", "unreachable", "resolution"]):
                break
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
def recommend_companies(student_profile):
    """
    Evaluates student profile suitability across 11 target employers.
    Uses Gemini API if available, falling back to local heuristic logic on failure.
    """
    cgpa = student_profile.get('cgpa', 7.0)
    coding = student_profile.get('coding', 50.0)
    aptitude = student_profile.get('aptitude', 50.0)
    comm = student_profile.get('communication', 50.0)
    tech_interview = student_profile.get('technical_interview', 50.0)
    projects = student_profile.get('projects', 1)
    internships = student_profile.get('internships', 0)
    certifications = student_profile.get('certifications', 1)
    languages = str(student_profile.get('languages', ''))
    
    prompt = f"""
You are an expert corporate recruitment advisor AI.
Analyze the following student profile metrics and evaluate their suitability fit across these 11 employers:
- Google (Tier 1 (Product Giant))
- Microsoft (Tier 1 (Product Giant))
- Amazon (Tier 1 (Product Giant))
- Zoho (Tier 2 (Core Product))
- IBM (Tier 2 (Global Technology))
- Accenture (Tier 3 (IT Consulting))
- Cognizant (Tier 3 (IT Services))
- Capgemini (Tier 3 (IT Services))
- TCS (Tier 3 (IT Services))
- Infosys (Tier 3 (IT Services))
- Wipro (Tier 3 (IT Services))

Student Profile:
- CGPA: {cgpa:.2f}/10.0
- Coding Competency: {coding:.1f}/100
- Aptitude Score: {aptitude:.1f}/100
- Communication: {comm:.1f}/100
- Technical Interview: {tech_interview:.1f}/100
- Number of Projects: {projects}
- Number of Internships: {internships}
- Number of Certifications: {certifications}
- Known Programming Languages / Technologies: {languages}

For each of the 11 employers, calculate a suitability score (0.0 to 100.0 as a float), determine a fit level ("Strong Match", "Moderate Match", or "Aspirational"), write a tailored "Hiring Suitability" explanation, and provide specific, actionable "Advisory" preparation advice for that company.

Respond STRICTLY in JSON format matching the schema below:
[
  {{
    "company": "Company Name",
    "tier": "Tier 1 (Product Giant) / Tier 2 (Core Product) / Tier 3 (IT Services) / Tier 3 (IT Consulting)",
    "score": 88.5,
    "fit": "Strong Match",
    "reasons": "Detailed explanation of suitability based on their profile. E.g., 'Requires high coding capability (min. 80) and strong technical communication. Your current profile scores are...'",
    "advice": "Actionable prep advice. E.g., 'Focus on advanced Data Structures & Algorithms, System Design basics, and explaining your projects in detail.'"
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
            if all(k in item for k in ('company', 'tier', 'score', 'fit', 'reasons', 'advice')):
                # Ensure correct types
                item['score'] = float(item['score'])
                validated_recs.append(item)
                
        if len(validated_recs) == 11:
            # Sort by score descending
            validated_recs = sorted(validated_recs, key=lambda x: x['score'], reverse=True)
            return validated_recs
        else:
            raise ValueError("Invalid number of companies returned from LLM")
            
    except Exception as e:
        # Fallback to local heuristic logic on any failure
        print(f"[RECS] Company recommendation API failed: {e}. Falling back to heuristic.")
        return _recommend_companies_fallback(student_profile)

def _recommend_companies_fallback(student_profile):
    cgpa = student_profile.get('cgpa', 7.0)
    coding = student_profile.get('coding', 50.0)
    aptitude = student_profile.get('aptitude', 50.0)
    comm = student_profile.get('communication', 50.0)
    tech_interview = student_profile.get('technical_interview', 50.0)
    projects = student_profile.get('projects', 1)
    internships = student_profile.get('internships', 0)
    certifications = student_profile.get('certifications', 1)
    
    companies = [
        # Tier 1 (FAANG / Product Giants)
        {'name': 'Google', 'tier': 'Tier 1 (Product Giant)', 'min_cgpa': 8.5, 'min_coding': 85, 'min_tech': 85, 'type': 'tier1'},
        {'name': 'Microsoft', 'tier': 'Tier 1 (Product Giant)', 'min_cgpa': 8.3, 'min_coding': 83, 'min_tech': 83, 'type': 'tier1'},
        {'name': 'Amazon', 'tier': 'Tier 1 (Product Giant)', 'min_cgpa': 8.0, 'min_coding': 80, 'min_tech': 80, 'type': 'tier1'},
        
        # Tier 2 (Top Product & Technology Labs)
        {'name': 'Zoho', 'tier': 'Tier 2 (Core Product)', 'min_cgpa': 7.5, 'min_coding': 70, 'min_tech': 70, 'type': 'tier2'},
        {'name': 'IBM', 'tier': 'Tier 2 (Global Technology)', 'min_cgpa': 7.2, 'min_coding': 65, 'min_tech': 65, 'type': 'tier2'},
        
        # Tier 3 (Global Service & IT Consulting)
        {'name': 'Accenture', 'tier': 'Tier 3 (IT Consulting)', 'min_cgpa': 6.5, 'min_coding': 50, 'min_tech': 50, 'type': 'tier3'},
        {'name': 'Cognizant', 'tier': 'Tier 3 (IT Services)', 'min_cgpa': 6.2, 'min_coding': 45, 'min_tech': 45, 'type': 'tier3'},
        {'name': 'Capgemini', 'tier': 'Tier 3 (IT Services)', 'min_cgpa': 6.2, 'min_coding': 45, 'min_tech': 45, 'type': 'tier3'},
        {'name': 'TCS', 'tier': 'Tier 3 (IT Services)', 'min_cgpa': 6.0, 'min_coding': 40, 'min_tech': 40, 'type': 'tier3'},
        {'name': 'Infosys', 'tier': 'Tier 3 (IT Services)', 'min_cgpa': 6.0, 'min_coding': 40, 'min_tech': 40, 'type': 'tier3'},
        {'name': 'Wipro', 'tier': 'Tier 3 (IT Services)', 'min_cgpa': 6.0, 'min_coding': 40, 'min_tech': 40, 'type': 'tier3'}
    ]
    
    recs = []
    
    for comp in companies:
        name = comp['name']
        tier_type = comp['type']
        
        # Calculate fit score based on requirements
        if tier_type == 'tier1':
            # High weight on Coding & Technical Interview
            score = (coding * 0.4) + (tech_interview * 0.3) + (cgpa * 10.0 * 0.2) + (internships * 5)
            score = min(100.0, max(20.0, score))
            
            # Preparation advice
            if score >= 80:
                fit = "Strong Match"
                advice = "You have an excellent profile. Focus on advanced Data Structures & Algorithms (Leetcode Medium/Hard), System Design basics, and explaining your projects in detail."
            elif score >= 60:
                fit = "Moderate Match"
                advice = "A decent match. Elevate your coding score (aim for 85+) and practice competitive coding. Complete an internship to boost your resume."
            else:
                fit = "Aspirational"
                advice = "Google/Microsoft/Amazon require strong problem-solving skills. Spend the next 3 months doing focused DSA prep, solve 150+ problems, and improve your CGPA/Technical Interview scores."
                
            reasons = f"Requires high coding capability (min. {comp['min_coding']}) and strong technical communication. Your current profile scores are: Coding: {coding:.1f}, CGPA: {cgpa:.2f}."
            
        elif tier_type == 'tier2':
            # Balance of Coding, Projects, and CGPA
            score = (coding * 0.3) + (tech_interview * 0.25) + (cgpa * 10.0 * 0.2) + (projects * 4) + (aptitude * 0.1)
            score = min(100.0, max(30.0, score))
            
            if score >= 75:
                fit = "Strong Match"
                advice = "Very high compatibility. Zoho focuses heavily on practical coding tests (app building) rather than standard MCQs. Practice building clean console or web apps in Java/Python/C++."
            elif score >= 55:
                fit = "Moderate Match"
                advice = "Good match. Practice coding, study database normalization, SQL queries, and object-oriented concepts."
            else:
                fit = "Aspirational"
                advice = "Work on building 2 solid projects and raise your coding score to at least 70. Learn fundamental DBMS and OOPS concepts."
                
            reasons = f"Requires good problem-solving foundation (min. {comp['min_coding']} coding) and project implementation capacity. Your Coding Score: {coding:.1f}, Projects: {projects}."
            
        else: # tier 3
            # Weight on Aptitude, Communication, and foundational coding
            score = (aptitude * 0.3) + (comm * 0.3) + (coding * 0.2) + (cgpa * 10 * 0.1) + (certifications * 2)
            score = min(100.0, max(40.0, score))
            
            if score >= 70:
                fit = "Strong Match"
                advice = "Excellent profile for service giants. You qualify for high-tier entry-level packages (like TCS Digital or Cognizant Next/Diff). Focus on communication, logical reasoning, and basic coding."
            elif score >= 50:
                fit = "Moderate Match"
                advice = "Clear fit. Revise quantitative aptitude, logical reasoning, and basic python/java coding concepts. Practice mock interviews."
            else:
                fit = "Aspirational"
                advice = "Improve your communication score and aptitude. Take online mock tests for TCS NQT / Accenture placement rounds."
                
            reasons = f"Requires basic aptitude (min. 50) and good verbal communication. Your Aptitude: {aptitude:.1f}, Communication: {comm:.1f}."
            
        recs.append({
            'company': name,
            'tier': comp['tier'],
            'score': round(score, 1),
            'fit': fit,
            'reasons': reasons,
            'advice': advice
        })
        
    # Sort by fit score (descending)
    recs = sorted(recs, key=lambda x: x['score'], reverse=True)
    return recs
