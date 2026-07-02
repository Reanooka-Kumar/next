import pandas as pd
import numpy as np

# Mapping of roles to standard technical skills required by industry
ROLE_SKILLS = {
    'Software Engineer': {
        'mandatory_langs': ['java', 'c++', 'python'],
        'conceptual_skills': ['Data Structures & Algorithms (DSA)', 'Object-Oriented Programming (OOPS)', 'Git & Version Control', 'System Design Basics'],
        'certs': [
            {'name': 'Oracle Certified Associate (Java)', 'prob_gain': 7.5},
            {'name': 'AWS Certified Developer Associate', 'prob_gain': 10.0},
            {'name': 'Meta Software Engineer Professional Certificate', 'prob_gain': 8.0}
        ]
    },
    'Data Scientist': {
        'mandatory_langs': ['python', 'sql'],
        'conceptual_skills': ['Probability & Statistics', 'Machine Learning Models', 'Data Analysis (Pandas/NumPy)', 'Data Visualization (Plotly/Matplotlib)', 'Model Evaluation'],
        'certs': [
            {'name': 'IBM Data Science Professional Certificate', 'prob_gain': 12.0},
            {'name': 'Google Advanced Data Analytics Certificate', 'prob_gain': 10.0},
            {'name': 'Microsoft Certified: Azure Data Scientist Associate', 'prob_gain': 15.0}
        ]
    },
    'Machine Learning Engineer': {
        'mandatory_langs': ['python', 'sql', 'c++'],
        'conceptual_skills': ['Machine Learning (Supervised/Unsupervised)', 'Deep Learning (TensorFlow/PyTorch)', 'Linear Algebra & Calculus', 'MLOps & Model Deployment', 'Git & Docker'],
        'certs': [
            {'name': 'DeepLearning.AI TensorFlow Developer Certificate', 'prob_gain': 15.0},
            {'name': 'AWS Certified Machine Learning - Specialty', 'prob_gain': 18.0},
            {'name': 'Google Cloud Professional ML Engineer', 'prob_gain': 15.0}
        ]
    },
    'Data Analyst': {
        'mandatory_langs': ['sql', 'python'],
        'conceptual_skills': ['Excel Data Modeling', 'BI Tools (Power BI / Tableau)', 'Descriptive Statistics', 'Data Cleaning & ETL', 'SQL Query Optimization'],
        'certs': [
            {'name': 'Google Data Analytics Professional Certificate', 'prob_gain': 10.0},
            {'name': 'Microsoft Power BI Data Analyst Associate (PL-300)', 'prob_gain': 12.0},
            {'name': 'Tableau Desktop Specialist', 'prob_gain': 8.0}
        ]
    },
    'Business Analyst': {
        'mandatory_langs': ['sql'],
        'conceptual_skills': ['Business Communication', 'Requirements Gathering & Agile', 'Excel Data Analysis', 'Process Flowcharting (Visio/Lucidchart)', 'BI Basics'],
        'certs': [
            {'name': 'IIBA Entry Certificate in Business Analysis (ECBA)', 'prob_gain': 12.0},
            {'name': 'Google Project Management Professional Certificate', 'prob_gain': 8.5},
            {'name': 'Microsoft Certified: Power Platform Functional Consultant', 'prob_gain': 10.0}
        ]
    },
    'AI Engineer': {
        'mandatory_langs': ['python'],
        'conceptual_skills': ['Natural Language Processing (NLP)', 'Generative AI & LLMs (OpenAI/LangChain)', 'Deep Learning', 'Computer Vision Basics', 'Cloud APIs Integration', 'Vector Databases'],
        'certs': [
            {'name': 'Microsoft Certified: Azure AI Engineer Associate', 'prob_gain': 16.0},
            {'name': 'Intel Edge AI Certification', 'prob_gain': 10.0},
            {'name': 'DeepLearning.AI Generative AI Developer', 'prob_gain': 14.0}
        ]
    },
    'Full Stack Developer': {
        'mandatory_langs': ['javascript', 'sql', 'python', 'java'],
        'conceptual_skills': ['Frontend Frameworks (React / Vue)', 'Backend Runtimes (Node.js / Django)', 'HTML5 & CSS3 Responsive UI', 'REST API Design & Web Security', 'NoSQL Databases (MongoDB)'],
        'certs': [
            {'name': 'Meta Front-End/Back-End Developer Certificate', 'prob_gain': 10.0},
            {'name': 'AWS Certified Developer Associate', 'prob_gain': 12.0},
            {'name': 'MongoDB Certified Developer Associate', 'prob_gain': 8.0}
        ]
    }
}

def analyze_skill_gap(student_profile, target_role):
    """
    Compares student profile against target career requirements.
    Identifies existing skills, missing skills, certification suggestions, 
    and estimated improvement in placement probability.
    """
    if target_role not in ROLE_SKILLS:
        target_role = 'Software Engineer' # Fallback
        
    requirements = ROLE_SKILLS[target_role]
    
    # 1. Existing Skills extraction
    student_langs = [l.strip().lower() for l in str(student_profile.get('languages', '')).split(',') if l.strip()]
    existing = []
    
    # Add matched programming languages
    for lang in student_langs:
        existing.append(lang.capitalize())
        
    # Add conceptual/soft-skill proxies
    coding_score = student_profile.get('coding', 50.0)
    comm_score = student_profile.get('communication', 50.0)
    projects = student_profile.get('projects', 1)
    certifications = student_profile.get('certifications', 1)
    
    if coding_score >= 70:
        existing.append("Basic Coding Logic")
    if coding_score >= 85:
        existing.append("Data Structures & Algorithms (DSA)")
    if comm_score >= 75:
        existing.append("Strong Verbal Communication")
    if projects >= 2:
        existing.append("Project Design & Implementation")
    if certifications >= 2:
        existing.append("Continuous Professional Learning")
        
    # 2. Missing Skills determination
    missing = []
    priority_level = "Medium"
    
    # Check mandatory languages
    missing_langs = []
    for req_lang in requirements['mandatory_langs']:
        if req_lang not in student_langs:
            missing_langs.append(req_lang.capitalize())
            
    if missing_langs:
        missing.append(f"Language Proficiency: {', '.join(missing_langs)}")
        priority_level = "High" # Missing a core language is high priority
        
    # Check conceptual skills
    for skill in requirements['conceptual_skills']:
        # If student has low coding, they miss DSA/System design
        if "Algorithms" in skill and coding_score < 75:
            missing.append(skill)
            priority_level = "High"
        elif "System Design" in skill and coding_score < 80:
            missing.append(skill)
        elif "Deep Learning" in skill and coding_score < 75:
            missing.append(skill)
            priority_level = "High" if target_role in ['Machine Learning Engineer', 'AI Engineer'] else "Medium"
        elif "BI Tools" in skill and target_role in ['Data Analyst', 'Business Analyst'] and projects < 2:
            missing.append(skill)
        elif "REST API" in skill and target_role == 'Full Stack Developer' and projects < 2:
            missing.append(skill)
        elif "Communication" in skill and comm_score < 70:
            missing.append(skill)
            priority_level = "High" if target_role == 'Business Analyst' else "Medium"
            
    # Add default missing skills if student is very beginner
    if not missing and len(existing) < 4:
        missing.append("General Project Development Experience")
        
    # 3. Recommended Certifications
    recommended_certs = []
    for cert in requirements['certs']:
        recommended_certs.append({
            'name': cert['name'],
            'prob_gain': cert['prob_gain']
        })
        
    # 4. Total Placement probability improvement estimation
    # Compiles improvement based on addressing gaps
    base_improvement = 0.0
    if priority_level == "High":
        base_improvement += 15.0
    else:
        base_improvement += 8.0
        
    # Increment for learning languages
    base_improvement += len(missing_langs) * 5.0
    # Increment for certifications
    base_improvement += min(2, len(recommended_certs)) * 3.5
    
    max_prob = student_profile.get('placement_probability', 50.0)
    estimated_gain = min(100.0 - max_prob, base_improvement)
    estimated_gain = max(2.0, round(estimated_gain, 1))
    
    return {
        'target_role': target_role,
        'existing_skills': list(set(existing)),
        'missing_skills': missing[:4] if missing else ["No significant skills gap! Keep practicing."],
        'recommended_certifications': recommended_certs,
        'priority_level': priority_level,
        'estimated_probability_gain': estimated_gain
    }
