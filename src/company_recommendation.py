import pandas as pd
import numpy as np

def recommend_companies(student_profile):
    """
    Evaluates student profile suitability across 11 target employers.
    Categorizes them by Tier and provides tailored insights.
    """
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
