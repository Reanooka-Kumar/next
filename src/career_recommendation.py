import pandas as pd
import numpy as np

def recommend_careers(student_profile):
    """
    Computes a match percentage and specific feedback for 7 career paths.
    student_profile: dict containing student raw inputs
    """
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
