import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define the features we will use for the models
NUMERICAL_COLS = [
    'Age', 'CGPA', 'Attendance', 'Aptitude Score', 'Coding Score', 
    'Communication Score', 'Technical Interview Score', 'Internships', 
    'Certifications', 'Projects Completed', 'Languages Count', 
    'Soft Skills Rating', 'Extra Curricular'
]

CATEGORICAL_COLS = ['Department']

DEPARTMENTS = [
    "Computer Science & Engineering",
    "Information Technology",
    "Electronics & Communication Engineering",
    "Electrical & Electronics Engineering",
    "Mechanical Engineering",
    "Civil Engineering"
]

def map_raw_df_to_features(df):
    """
    Transforms raw dataframe columns into features suitable for ML modeling.
    Supports both training dataset and validation uploads.
    """
    processed = pd.DataFrame(index=df.index)
    
    # 1. Name and ID (pass-through / metadata)
    if 'Name' in df.columns:
        processed['Name'] = df['Name']
    else:
        processed['Name'] = 'Student'
        
    if 'StudentId' in df.columns:
        processed['StudentId'] = df['StudentId']
        
    # 2. Numerical Mapping
    processed['Age'] = df['Age'].astype(float)
    processed['CGPA'] = df['CGPA'].astype(float)
    processed['Attendance'] = df['Attendance'].astype(float)
    processed['Aptitude Score'] = df['Aptitude Score'].astype(float)
    processed['Coding Score'] = df['Coding Score'].astype(float)
    processed['Technical Interview Score'] = df['Technical Interview Score'].astype(float)
    
    # Certifications
    if 'Certifications' in df.columns:
        processed['Certifications'] = df['Certifications'].astype(float)
    elif 'Workshops/Certificatios' in df.columns:
        processed['Certifications'] = df['Workshops/Certificatios'].astype(float)
    else:
        processed['Certifications'] = 0.0
        
    # Projects Completed: sum of Major & Mini Projects if individual columns exist, or directly
    if 'Projects Completed' in df.columns:
        processed['Projects Completed'] = df['Projects Completed'].astype(float)
    else:
        major = df['Major Projects'].astype(float) if 'Major Projects' in df.columns else 0.0
        mini = df['Mini Projects'].astype(float) if 'Mini Projects' in df.columns else 0.0
        processed['Projects Completed'] = major + mini
        
    # Technical Skills Rating
    if 'Technical Skills Rating' in df.columns:
        processed['Technical Skills Rating'] = df['Technical Skills Rating'].astype(float)
    elif 'Skills' in df.columns:
        processed['Technical Skills Rating'] = df['Skills'].astype(float)
    else:
        processed['Technical Skills Rating'] = 5.0
        
    # Communication Score
    if 'Communication Score' in df.columns:
        processed['Communication Score'] = df['Communication Score'].astype(float)
    elif 'Communication Skill Rating' in df.columns:
        # Scale if it's in the old 1-5 scale
        processed['Communication Score'] = (df['Communication Skill Rating'] * 20).astype(float)
    else:
        processed['Communication Score'] = 75.0
        
    # Soft Skills Rating
    processed['Soft Skills Rating'] = df['Soft Skills Rating'].astype(float)
    
    # 10th & 12th Percentages and Backlogs
    processed['12th Percentage'] = df['12th Percentage'].astype(float) if '12th Percentage' in df.columns else 70.0
    processed['10th Percentage'] = df['10th Percentage'].astype(float) if '10th Percentage' in df.columns else 70.0
    processed['backlogs'] = df['backlogs'].astype(float) if 'backlogs' in df.columns else 0.0
    
    # 3. Binary Mappings (Yes/No -> 1/0)
    processed['Internships'] = df['Internship'].map({'Yes': 1.0, 'No': 0.0}).fillna(0.0) if 'Internship' in df.columns else 0.0
    # Also support numeric or binary representation
    if 'Internships' not in processed.columns and 'Internship Experience' in df.columns:
        # If it is numeric
        if df['Internship Experience'].dtype in [np.int64, np.float64]:
            processed['Internships'] = df['Internship Experience'].astype(float)
        else:
            processed['Internships'] = df['Internship Experience'].map({'Yes': 1.0, 'No': 0.0}).fillna(0.0)
            
    processed['Hackathon'] = df['Hackathon'].map({'Yes': 1.0, 'No': 0.0}).fillna(0.0) if 'Hackathon' in df.columns else 0.0
    processed['Extra Curricular'] = df['Extra Curricular Activities'].map({'Yes': 1.0, 'No': 0.0}).fillna(0.0) if 'Extra Curricular Activities' in df.columns else 0.0
    
    # 4. Programming Languages Count
    if 'Programming Languages Known' in df.columns:
        processed['Languages Count'] = df['Programming Languages Known'].apply(
            lambda x: len([l.strip() for l in str(x).split(',') if l.strip()]) if pd.notnull(x) else 0.0
        ).astype(float)
    else:
        processed['Languages Count'] = 1.0
        
    # 5. Department Categorical Mapping
    processed['Department'] = df['Department'].fillna(DEPARTMENTS[0])
    
    # 6. Target Labels (if present in df)
    if 'PlacementStatus' in df.columns:
        # support both Placed/NotPlaced and 1/0
        if df['PlacementStatus'].dtype == object:
            processed['PlacementStatus'] = df['PlacementStatus'].map({'Placed': 1, 'NotPlaced': 0, 'Not Placed': 0}).fillna(0)
        else:
            processed['PlacementStatus'] = df['PlacementStatus'].astype(int)
            
    if 'Expected Salary' in df.columns:
        processed['Expected Salary'] = df['Expected Salary'].astype(float)
        
    return processed

def prepare_ml_data(features_df, fit_scaler=None):
    """
    Encodes and scales features.
    If fit_scaler is provided, it will use that scaler to transform the numerical data.
    If None, it fits a new StandardScaler.
    Returns: X (scaled numpy array), y (labels or None), scaler, feature_names list
    """
    # Create copy to avoid modifying original
    df = features_df.copy()
    
    # Perform One-Hot Encoding on Department
    dept_dummies = pd.get_dummies(df['Department'], prefix='Dept')
    
    # Align department columns with standard list to avoid missing columns
    for dept in DEPARTMENTS:
        col_name = f"Dept_{dept}"
        if col_name not in dept_dummies.columns:
            dept_dummies[col_name] = 0
            
    # Keep columns in exact order
    dept_cols = [f"Dept_{dept}" for dept in DEPARTMENTS]
    dept_dummies = dept_dummies[dept_cols]
    
    # Combine numerical features and department dummies
    X_num = df[NUMERICAL_COLS]
    
    # Fit or apply scaler
    if fit_scaler is None:
        scaler = StandardScaler()
        X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=NUMERICAL_COLS, index=df.index)
    else:
        scaler = fit_scaler
        X_num_scaled = pd.DataFrame(scaler.transform(X_num), columns=NUMERICAL_COLS, index=df.index)
        
    X_final = pd.concat([X_num_scaled, dept_dummies], axis=1)
    feature_names = X_final.columns.tolist()
    
    y = df['PlacementStatus'].values if 'PlacementStatus' in df.columns else None
    
    return X_final.values, y, scaler, feature_names

def preprocess_single_student(student_dict, scaler):
    """
    Preprocesses a single student profile dictionary.
    Returns a numpy array of shape (1, n_features) and the raw feature dictionary.
    """
    # 1. Create a 1-row DataFrame
    raw_df = pd.DataFrame([student_dict])
    
    # Convert 'Internship Experience' to 'Internship' and other minor translations
    if 'internships' in student_dict:
        raw_df['Internship'] = 'Yes' if student_dict['internships'] > 0 else 'No'
    if 'certifications' in student_dict:
        raw_df['Certifications'] = student_dict['certifications']
    if 'projects' in student_dict:
        # simulate major/mini
        raw_df['Major Projects'] = min(student_dict['projects'], 2)
        raw_df['Mini Projects'] = max(student_dict['projects'] - 2, 0)
    if 'languages' in student_dict:
        raw_df['Programming Languages Known'] = student_dict['languages']
    if 'extra_curricular' in student_dict:
        raw_df['Extra Curricular Activities'] = student_dict['extra_curricular']
    if 'soft_skills' in student_dict:
        raw_df['Soft Skills Rating'] = student_dict['soft_skills']
    if 'technical_interview' in student_dict:
        raw_df['Technical Interview Score'] = student_dict['technical_interview']
    if 'communication' in student_dict:
        raw_df['Communication Score'] = student_dict['communication']
    if 'coding' in student_dict:
        raw_df['Coding Score'] = student_dict['coding']
    if 'aptitude' in student_dict:
        raw_df['Aptitude Score'] = student_dict['aptitude']
    if 'attendance' in student_dict:
        raw_df['Attendance'] = student_dict['attendance']
    if 'cgpa' in student_dict:
        raw_df['CGPA'] = student_dict['cgpa']
    if 'age' in student_dict:
        raw_df['Age'] = student_dict['age']
    if 'department' in student_dict:
        raw_df['Department'] = student_dict['department']
        
    # Pass raw dataframe directly to features mapper
                
    features_df = map_raw_df_to_features(raw_df)
    
    # 2. Prepare ML data (using the fitted scaler)
    X, _, _, feature_names = prepare_ml_data(features_df, fit_scaler=scaler)
    
    # Extract the mapped numeric values for explanation purposes
    mapped_features_dict = {}
    for col in NUMERICAL_COLS:
        mapped_features_dict[col] = features_df[col].values[0]
    mapped_features_dict['Department'] = features_df['Department'].values[0]
    
    return X, mapped_features_dict, feature_names
