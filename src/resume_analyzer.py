import re
from pypdf import PdfReader

# Master skill keywords database categorized
SKILL_KEYWORDS = {
    'programming': ['python', 'java', 'c++', 'javascript', 'html', 'css', 'typescript', 'go', 'kotlin', 'swift', 'rust', 'ruby', 'c#', 'php', 'sql'],
    'data_science': ['machine learning', 'deep learning', 'statistics', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'keras', 'pytorch', 'data analytics', 'data science', 'nlp', 'natural language processing', 'computer vision', 'r programming', 'tableau', 'power bi'],
    'databases': ['mysql', 'postgresql', 'sqlite', 'mongodb', 'redis', 'oracle', 'nosql', 'dbms'],
    'developer_tools': ['git', 'github', 'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'linux', 'api', 'flask', 'django', 'react', 'node.js', 'express', 'spring boot']
}

def extract_text_from_pdf(pdf_file):
    """
    Extracts text content from a PDF file using pypdf.
    pdf_file can be a path or a file-like object (e.g., BytesIO from Streamlit file upload).
    """
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def analyze_resume(resume_text, target_role):
    """
    Analyzes resume text for ATS scoring, keyword matches, and improvement tips.
    """
    if not resume_text or len(resume_text.strip()) == 0:
        return {
            'ats_score': 0,
            'skills_found': [],
            'missing_keywords': [],
            'sections_found': {},
            'suggestions': ["No resume text detected. Please ensure the PDF is not an image scan."]
        }
        
    text_lower = resume_text.lower()
    
    # 1. Section Checks (case-insensitive search)
    sections = {
        'Education': ['education', 'academic', 'qualifications', 'degree'],
        'Experience': ['experience', 'employment', 'work history', 'internship', 'professional experience'],
        'Projects': ['project', 'projects', 'academic projects', 'key projects'],
        'Skills': ['skills', 'technical skills', 'core competencies', 'expertise', 'languages'],
        'Certifications': ['certification', 'certifications', 'credentials', 'workshops']
    }
    
    sections_found = {}
    section_score = 0
    for section_name, keywords in sections.items():
        found = False
        for kw in keywords:
            # Match word boundaries or distinct headers
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                found = True
                break
        sections_found[section_name] = found
        if found:
            section_score += 6 # Max 30 points for sections (5 sections * 6)
            
    # 2. Extract Technical Skills Found
    skills_found = []
    for category, kws in SKILL_KEYWORDS.items():
        for kw in kws:
            pattern = r'\b' + re.escape(kw) + r'\b'
            # Custom handling for C++ which has special characters
            if kw == 'c++':
                if 'c++' in text_lower or 'cpp' in text_lower:
                    skills_found.append('C++')
                    continue
            if re.search(pattern, text_lower):
                skills_found.append(kw.capitalize() if kw not in ['sql', 'aws', 'gcp', 'nlp', 'dbms', 'api'] else kw.upper())
                
    skills_found = list(set(skills_found))
    
    # Skills score: 4 points per skill, max 40 points
    skills_score = min(40, len(skills_found) * 4)
    
    # 3. Determine Missing Keywords based on Target Role
    role_kws = {
        'Software Engineer': ['DSA', 'Java', 'C++', 'Git', 'System Design', 'OOPs'],
        'Data Scientist': ['Python', 'SQL', 'Machine Learning', 'Pandas', 'Statistics', 'Tableau'],
        'Machine Learning Engineer': ['Python', 'PyTorch', 'TensorFlow', 'MLOps', 'Deep Learning', 'Git'],
        'Data Analyst': ['SQL', 'Excel', 'Power BI', 'Tableau', 'Statistics', 'Python'],
        'Business Analyst': ['SQL', 'Excel', 'Agile', 'Requirements', 'Tableau', 'Scrum'],
        'AI Engineer': ['Python', 'Generative AI', 'LLMs', 'NLP', 'Deep Learning', 'Docker'],
        'Full Stack Developer': ['JavaScript', 'React', 'Node.js', 'SQL', 'Git', 'HTML', 'CSS']
    }
    
    target_role_kws = role_kws.get(target_role, ['Python', 'Git', 'SQL'])
    missing_keywords = []
    
    for kw in target_role_kws:
        kw_lower = kw.lower()
        # Handle custom matches
        if kw_lower == 'dsa':
            matched = 'data structures' in text_lower or 'dsa' in text_lower or 'algorithms' in text_lower
        elif kw_lower == 'oops':
            matched = 'oop' in text_lower or 'oops' in text_lower or 'object oriented' in text_lower
        elif kw_lower == 'llms':
            matched = 'llm' in text_lower or 'large language' in text_lower or 'gpt' in text_lower
        elif kw_lower == 'git':
            matched = 'git' in text_lower or 'github' in text_lower
        elif kw_lower == 'c++':
            matched = 'c++' in text_lower or 'cpp' in text_lower
        else:
            pattern = r'\b' + re.escape(kw_lower) + r'\b'
            matched = re.search(pattern, text_lower) is not None
            
        if not matched:
            missing_keywords.append(kw)
            
    # Missing keywords penalty: subtract 4 points per missing keyword from a base of 20
    keyword_score = max(0, 20 - (len(missing_keywords) * 4))
    
    # 4. Length and Formatting Heuristics
    words = text_lower.split()
    word_count = len(words)
    
    formatting_score = 10
    length_warning = None
    
    if word_count < 150:
        formatting_score = 3  # Way too short
        length_warning = "Your resume is extremely short (under 150 words). Recruiters prefer detailed profiles."
    elif word_count < 300:
        formatting_score = 6  # Slightly short
        length_warning = "Your resume is a bit short. Add more bullet points detailing your projects and certifications."
    elif word_count > 1200:
        formatting_score = 5  # Too long, potentially multi-page/cluttered
        length_warning = "Your resume is quite long (over 1200 words). Try to condense it to a concise 1 or 2-page format."
        
    # ATS score calculation (out of 100)
    # Sections (30) + Skills (40) + Target Keywords (20) + Formatting/Length (10)
    ats_score = section_score + skills_score + keyword_score + formatting_score
    ats_score = min(100, max(15, ats_score))
    
    # 5. Suggestions for improvement
    suggestions = []
    
    # Section feedback
    missing_sections = [sec for sec, found in sections_found.items() if not found]
    if missing_sections:
        suggestions.append(f"Add the following missing sections to your resume: **{', '.join(missing_sections)}**.")
        
    # Keyword suggestions
    if missing_keywords:
        suggestions.append(f"Incorporate missing core keywords for the **{target_role}** role: **{', '.join(missing_keywords)}**.")
        
    # Formatting suggestions
    if length_warning:
        suggestions.append(length_warning)
        
    if not sections_found['Certifications']:
        suggestions.append("You do not have a dedicated 'Certifications' section. Add standard certifications to stand out to recruiters.")
        
    if len(skills_found) < 5:
        suggestions.append("List more specific technical tools, frameworks, and databases rather than only general programming languages.")
        
    if len(suggestions) == 0:
        suggestions.append("Your resume looks clean and satisfies ATS parsing parameters. Highlight key project metrics next.")
        
    return {
        'ats_score': int(ats_score),
        'skills_found': skills_found,
        'missing_keywords': missing_keywords,
        'sections_found': sections_found,
        'suggestions': suggestions,
        'word_count': word_count
    }
