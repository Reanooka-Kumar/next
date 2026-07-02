import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

DB_NAME = "placement_history.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            name TEXT,
            age INTEGER,
            department TEXT,
            cgpa REAL,
            attendance REAL,
            aptitude REAL,
            coding REAL,
            communication REAL,
            technical_interview REAL,
            internships INTEGER,
            certifications INTEGER,
            projects INTEGER,
            languages TEXT,
            soft_skills REAL,
            extra_curricular TEXT,
            prediction TEXT,
            probability REAL,
            expected_salary REAL
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(data, prediction, probability, expected_salary):
    """
    Saves a student prediction record to the SQLite database.
    data: dict containing the student profile inputs
    prediction: 'Placed' or 'Not Placed'
    probability: float value (0 to 100)
    expected_salary: predicted package in LPA (float)
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO prediction_history (
            timestamp, name, age, department, cgpa, attendance, 
            aptitude, coding, communication, technical_interview, 
            internships, certifications, projects, languages, 
            soft_skills, extra_curricular, prediction, probability, expected_salary
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        data.get('name', 'N/A'),
        int(data.get('age', 21)),
        data.get('department', 'N/A'),
        float(data.get('cgpa', 0.0)),
        float(data.get('attendance', 0.0)),
        float(data.get('aptitude', 0.0)),
        float(data.get('coding', 0.0)),
        float(data.get('communication', 0.0)),
        float(data.get('technical_interview', 0.0)),
        int(data.get('internships', 0)),
        int(data.get('certifications', 0)),
        int(data.get('projects', 0)),
        data.get('languages', 'N/A'),
        float(data.get('soft_skills', 0.0)),
        data.get('extra_curricular', 'No'),
        prediction,
        float(probability),
        float(expected_salary)
    ))
    
    conn.commit()
    conn.close()

def get_prediction_history():
    """
    Retrieves all history records from the database.
    """
    init_db()
    conn = get_db_connection()
    query = "SELECT * FROM prediction_history ORDER BY id DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def search_history(search_query):
    """
    Search history records by student name or department.
    """
    init_db()
    conn = get_db_connection()
    query = """
        SELECT * FROM prediction_history 
        WHERE name LIKE ? OR department LIKE ?
        ORDER BY id DESC
    """
    df = pd.read_sql_query(query, conn, params=(f"%{search_query}%", f"%{search_query}%"))
    conn.close()
    return df

def clear_history():
    """
    Clears all prediction records.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history")
    conn.commit()
    conn.close()

def validate_csv_data(df):
    """
    Validates uploaded CSV data structure and values.
    Returns: (is_valid, report_message, cleaned_df)
    """
    required_cols = {
        'Name', 'Age', 'Department', 'CGPA', 'Attendance', 
        'Aptitude Score', 'Coding Score', 'Communication Score', 
        'Technical Interview Score', 'Internship', 'Certifications', 
        'Projects Completed', 'Programming Languages Known', 
        'Soft Skills Rating', 'Extra Curricular Activities'
    }
    
    # Check if placement status is in CSV (useful for model training update, but optional for prediction)
    cols = set(df.columns)
    missing_cols = required_cols - cols
    
    if len(missing_cols) > 0:
        return False, f"Missing required columns: {', '.join(missing_cols)}", df
        
    cleaned_df = df.copy()
    warnings = []
    
    # 1. Check for missing values and fill them
    null_counts = cleaned_df[list(required_cols)].isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            warnings.append(f"Imputed {count} missing values in '{col}'")
            if cleaned_df[col].dtype in [np.float64, np.int64]:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'N/A')
                
    # 2. Check value range constraints and clip if out of bounds
    # CGPA (0 to 10)
    out_cgpa = ((cleaned_df['CGPA'] < 0.0) | (cleaned_df['CGPA'] > 10.0)).sum()
    if out_cgpa > 0:
        warnings.append(f"Clipped {out_cgpa} values in 'CGPA' to range [0.0, 10.0]")
        cleaned_df['CGPA'] = np.clip(cleaned_df['CGPA'], 0.0, 10.0)
        
    # Scores (0 to 100)
    score_cols = ['Attendance', 'Aptitude Score', 'Coding Score', 'Communication Score', 'Technical Interview Score']
    for col in score_cols:
        out_score = ((cleaned_df[col] < 0.0) | (cleaned_df[col] > 100.0)).sum()
        if out_score > 0:
            warnings.append(f"Clipped {out_score} values in '{col}' to range [0.0, 100.0]")
            cleaned_df[col] = np.clip(cleaned_df[col], 0.0, 100.0)
            
    # Age (18 to 30)
    out_age = ((cleaned_df['Age'] < 18) | (cleaned_df['Age'] > 30)).sum()
    if out_age > 0:
        warnings.append(f"Clipped {out_age} values in 'Age' to range [18, 30]")
        cleaned_df['Age'] = np.clip(cleaned_df['Age'], 18, 30)
        
    # Ratings (1 to 5)
    out_soft = ((cleaned_df['Soft Skills Rating'] < 1.0) | (cleaned_df['Soft Skills Rating'] > 5.0)).sum()
    if out_soft > 0:
        warnings.append(f"Clipped {out_soft} values in 'Soft Skills Rating' to range [1.0, 5.0]")
        cleaned_df['Soft Skills Rating'] = np.clip(cleaned_df['Soft Skills Rating'], 1.0, 5.0)

    # Experience & Certifications (0 to 10)
    for col in ['Certifications', 'Projects Completed']:
        out_val = ((cleaned_df[col] < 0) | (cleaned_df[col] > 15)).sum()
        if out_val > 0:
            warnings.append(f"Clipped {out_val} values in '{col}' to range [0, 15]")
            cleaned_df[col] = np.clip(cleaned_df[col], 0, 15)
            
    report = "Data is valid. "
    if len(warnings) > 0:
        report += "Cleaned data with the following corrections:\n- " + "\n- ".join(warnings)
    else:
        report += "All checks passed with no warnings."
        
    return True, report, cleaned_df

# Run database init
init_db()
