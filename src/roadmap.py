import pandas as pd
import numpy as np

def generate_roadmap(student_profile, target_role):
    """
    Generates a personalized 12-week learning roadmap based on the student's
    weakest scores and chosen career role.
    """
    coding = student_profile.get('coding', 50.0)
    aptitude = student_profile.get('aptitude', 50.0)
    comm = student_profile.get('communication', 50.0)
    tech_interview = student_profile.get('technical_interview', 50.0)
    languages = str(student_profile.get('languages', '')).lower()
    
    # 1. Identify weak areas
    weak_coding = coding < 65
    weak_aptitude = aptitude < 65
    weak_comm = comm < 70
    weak_interview = tech_interview < 65
    needs_sql = "sql" not in languages
    
    roadmap = []
    
    # -------------------------------------------------------------
    # Weeks 1-2: Programming & Logic Foundation
    # -------------------------------------------------------------
    w1_2_topic = "Language Fundamentals & Basic Coding Logic"
    w1_2_focus = []
    if weak_coding:
        w1_2_focus.append("Solve 15 basic coding problems (arrays, strings, loops) daily on LeetCode/HackerRank.")
        w1_2_focus.append("Master syntax and standard libraries of your primary programming language (Python/Java/C++).")
    else:
        w1_2_focus.append("Revise object-oriented programming concepts (OOPs) in depth.")
        w1_2_focus.append("Learn language-specific performance tuning and memory management.")
        
    if needs_sql:
        w1_2_focus.append("Learn basic SQL keywords: SELECT, WHERE, GROUP BY, ORDER BY, and aggregates.")
        
    roadmap.append({
        'weeks': 'Weeks 1 - 2',
        'topic': w1_2_topic,
        'description': "Establish syntactic fluency and build standard logic structures. This phase removes friction in writing programs.",
        'action_items': w1_2_focus,
        'resources': "HackerRank (Easy Tracks), W3Schools (SQL), GeeksforGeeks (OOPs)"
    })
    
    # -------------------------------------------------------------
    # Weeks 3-4: Data Structures & Core Database Design
    # -------------------------------------------------------------
    w3_4_topic = "Data Structures & Relational Databases"
    w3_4_focus = [
        "Master Linear Data Structures: Linked Lists, Stacks, Queues, and Hash Maps.",
        "Solve 20 medium-level problems on Hash Maps and Arrays."
    ]
    if needs_sql or target_role in ['Data Scientist', 'Data Analyst', 'Business Analyst', 'Full Stack Developer']:
        w3_4_focus.append("Learn advanced SQL: JOINS, Subqueries, CTEs, and Window Functions.")
        w3_4_focus.append("Design schema tables and understand normalization (1NF, 2NF, 3NF).")
    else:
        w3_4_focus.append("Practice Binary Search and Sorting algorithm implementations.")
        
    roadmap.append({
        'weeks': 'Weeks 3 - 4',
        'topic': w3_4_topic,
        'description': "Develop efficiency in processing and storing data. Relational query skills are heavily tested in written rounds.",
        'action_items': w3_4_focus,
        'resources': "LeetCode (Arrays & HashMaps), SQLZoo, Mode Analytics SQL Tutorial"
    })
    
    # -------------------------------------------------------------
    # Weeks 5-6: Specialty Foundations (Role Specific)
    # -------------------------------------------------------------
    if target_role in ['Data Scientist', 'Machine Learning Engineer', 'AI Engineer']:
        w5_6_topic = "Math Foundations & Classical Machine Learning"
        w5_6_focus = [
            "Revise Linear Algebra (Matrices, Vectors) and Probability (Bayes Theorem, PDF/CDF).",
            "Learn classical supervised ML models: Linear/Logistic Regression, Decision Trees, Random Forests.",
            "Build simple regression and classification pipelines using Scikit-Learn."
        ]
        resources = "Kaggle Learn (Intro to ML), StatQuest YouTube Channel, Scikit-Learn Docs"
    elif target_role in ['Data Analyst', 'Business Analyst']:
        w5_6_topic = "Statistical Analysis & Business Intelligence"
        w5_6_focus = [
            "Learn Descriptive and Inferential Statistics (T-tests, hypothesis testing, correlation).",
            "Master Excel formulas, pivot tables, and VBA basics.",
            "Install Power BI / Tableau and learn to import data and connect sheets."
        ]
        resources = "Coursera Excel Skills for Business, Power BI Learn Pathway"
    else: # Software Engineer / Full Stack
        w5_6_topic = "Web Foundations & Advanced Data Structures"
        w5_6_focus = [
            "Master Non-Linear Data Structures: Binary Trees, Binary Search Trees, and Heaps.",
            "Learn frontend fundamentals: HTML5, CSS3, and modern ES6+ JavaScript.",
            "Understand HTTP methods, request/response cycles, and JSON data formats."
        ]
        resources = "MDN Web Docs, freeCodeCamp, LeetCode (Trees)"
        
    roadmap.append({
        'weeks': 'Weeks 5 - 6',
        'topic': w5_6_topic,
        'description': f"Deep-dive into core analytical and structural concepts for the {target_role} track.",
        'action_items': w5_6_focus,
        'resources': resources
    })
    
    # -------------------------------------------------------------
    # Weeks 7-8: Project Implementation & Building Portfolio
    # -------------------------------------------------------------
    w7_8_topic = "Hands-on Project Development & Git"
    w7_8_focus = [
        "Create a GitHub repository and learn version control: branching, committing, pushing, and pull requests.",
        "Develop project codebase (e.g. ML model pipeline, Full-stack app, BI dashboard)."
    ]
    if target_role in ['Data Scientist', 'Machine Learning Engineer', 'AI Engineer']:
        w7_8_focus.append("Build a prediction pipeline project: Fetch data, clean, train model, evaluate, and save model assets.")
    elif target_role in ['Data Analyst', 'Business Analyst']:
        w7_8_focus.append("Build an interactive dashboard tracking real-world datasets (e.g. Sales, COVID, Housing) with clear filters and KPI cards.")
    else:
        w7_8_focus.append("Build a REST API application using Node.js/Express or Python/FastAPI connected to a SQLite database.")
        
    roadmap.append({
        'weeks': 'Weeks 7 - 8',
        'topic': w7_8_topic,
        'description': "Apply theoretical learning by building a practical portfolio project. Recruiters hire based on what you build.",
        'action_items': w7_8_focus,
        'resources': "GitHub Guides, Kaggle Datasets, Traversy Media YouTube tutorials"
    })
    
    # -------------------------------------------------------------
    # Weeks 9-10: Advanced Topics & Quantitative Aptitude
    # -------------------------------------------------------------
    w9_10_topic = "Advanced Role Study & Quantitative Aptitude"
    w9_10_focus = []
    if weak_aptitude:
        w9_10_focus.append("Solve quantitative aptitude modules: Percentages, Profit/Loss, Time-Speed-Distance, Averages.")
        w9_10_focus.append("Practice logical reasoning puzzles: Syllogisms, Blood Relations, Seating Arrangements.")
    else:
        w9_10_focus.append("Solve 15 advanced aptitude questions daily to maintain speed.")
        
    if target_role == 'AI Engineer':
        w9_10_focus.append("Study LLM APIs, prompt engineering patterns, and retrieval-augmented generation (RAG) with LangChain.")
    elif target_role == 'Machine Learning Engineer':
        w9_10_focus.append("Learn Deep Learning basics (Neural Networks) and model deployment using Flask/Docker.")
    elif target_role == 'Software Engineer':
        w9_10_focus.append("Understand fundamental System Design: horizontal/vertical scaling, caching, load balancers.")
    else:
        w9_10_focus.append("Learn to structure business case studies and write user stories/agile requirements.")
        
    roadmap.append({
        'weeks': 'Weeks 9 - 10',
        'topic': w9_10_topic,
        'description': "Prepare for written screening tests which heavily filter candidates on aptitude and reasoning.",
        'action_items': w9_10_focus,
        'resources': "IndiaBIX (Aptitude), LeetCode System Design, DeepLearning.AI Prompt courses"
    })
    
    # -------------------------------------------------------------
    # Weeks 11-12: Mock Interviews & Final Polish
    # -------------------------------------------------------------
    w11_12_topic = "Mock Technical Interviews & Resume Refinement"
    w11_12_focus = [
        "Optimize your resume: Align keywords with target job descriptions and showcase projects.",
        "Conduct at least 3 mock technical interviews with peers or mentors."
    ]
    if weak_comm:
        w11_12_focus.append("Practice describing your project using the STAR method (Situation, Task, Action, Result) in English.")
        w11_12_focus.append("Record yourself answering standard HR questions (e.g. 'Tell me about yourself') and refine eye contact and pacing.")
    else:
        w11_12_focus.append("Prepare questions for interviewer and practice soft skills.")
        
    if weak_interview:
        w11_12_focus.append("Revise standard CS questions: DBMS (SQL queries), Operating Systems, Computer Networks.")
        
    roadmap.append({
        'weeks': 'Weeks 11 - 12',
        'topic': w11_12_topic,
        'description': "Transition from knowledge acquisition to performance optimization. Polish delivery, confidence, and speed.",
        'action_items': w11_12_focus,
        'resources': "Pramp (Mock Interviews), Resume Worded (ATS optimizer), interviewbit.com"
    })
    
    return roadmap
