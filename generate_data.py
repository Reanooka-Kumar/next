import pandas as pd
import numpy as np
import os

def enrich_dataset():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    input_file = "Placement_Prediction_data.csv"
    output_file = "Placement_Data_Enriched.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Cannot generate enriched dataset.")
        return
        
    df = pd.read_csv(input_file)
    print(f"Loaded original dataset with shape: {df.shape}")
    
    # 1. Names Generation
    first_names = [
        "Aarav", "Vihaan", "Aditya", "Sai", "Arjun", "Krishna", "Rohan", "Kabir", "Aryan", "Ishaan",
        "Priya", "Ananya", "Diya", "Sanya", "Riya", "Aadhya", "Siddhi", "Kavya", "Meera", "Neha",
        "Rahul", "Amit", "Sanjay", "Karan", "Nikhil", "Sneha", "Tanvi", "Pooja", "Vikram", "Abhishek",
        "Deepak", "Vivek", "Divya", "Kiran", "Meghna", "Aditi", "Harsh", "Yash", "Dev", "Alok"
    ]
    last_names = [
        "Sharma", "Patel", "Verma", "Rao", "Iyer", "Nair", "Singh", "Gupta", "Kumar", "Reddy",
        "Choudhury", "Joshi", "Mehta", "Mishra", "Pandey", "Sen", "Das", "Banerjee", "Chatterjee", "Pillai"
    ]
    
    n_rows = len(df)
    names = []
    for _ in range(n_rows):
        first = np.random.choice(first_names)
        last = np.random.choice(last_names)
        names.append(f"{first} {last}")
    df['Name'] = names
    
    # 2. Age (20 - 24)
    df['Age'] = np.random.randint(20, 25, size=n_rows)
    
    # 3. Department
    departments = [
        "Computer Science & Engineering",
        "Information Technology",
        "Electronics & Communication Engineering",
        "Electrical & Electronics Engineering",
        "Mechanical Engineering",
        "Civil Engineering"
    ]
    # CS (35%), IT (20%), ECE (20%), EEE (10%), Mech (10%), Civil (5%)
    dept_probs = [0.35, 0.20, 0.20, 0.10, 0.10, 0.05]
    df['Department'] = np.random.choice(departments, size=n_rows, p=dept_probs)
    
    # 4. Placement Status mapping helper
    # In original dataset, PlacementStatus is 'Placed' / 'NotPlaced'
    placed_mask = df['PlacementStatus'] == 'Placed'
    
    # 5. Attendance (%)
    # Placed students will have slightly higher attendance on average
    attendance = np.zeros(n_rows)
    attendance[placed_mask] = np.random.uniform(75, 99, size=placed_mask.sum())
    attendance[~placed_mask] = np.random.uniform(60, 90, size=(~placed_mask).sum())
    # Add minor noise
    noise = np.random.normal(0, 3, size=n_rows)
    df['Attendance'] = np.clip(attendance + noise, 55.0, 100.0).round(1)
    
    # 6. Aptitude Score (30 - 100)
    aptitude = np.zeros(n_rows)
    aptitude[placed_mask] = np.random.uniform(70, 100, size=placed_mask.sum())
    aptitude[~placed_mask] = np.random.uniform(30, 75, size=(~placed_mask).sum())
    noise = np.random.normal(0, 5, size=n_rows)
    df['Aptitude Score'] = np.clip(aptitude + noise, 30.0, 100.0).round(1)
    
    # 7. Coding Score (30 - 100)
    # Correlated with CGPA, Skills (original skills rating 1-10), Department and Placement Status
    coding = np.zeros(n_rows)
    for idx, row in df.iterrows():
        is_placed = 1 if row['PlacementStatus'] == 'Placed' else 0
        dept = row['Department']
        cgpa_val = row['CGPA']
        skills_val = row['Skills']
        
        # Base coding score
        base = 40
        # Boost for CS/IT
        if dept in ["Computer Science & Engineering", "Information Technology"]:
            base += 15
        elif dept == "Electronics & Communication Engineering":
            base += 5
            
        # Boost for CGPA and Skills
        base += (cgpa_val - 6.5) * 10
        base += (skills_val - 5) * 4
        
        # Boost for Placed
        if is_placed == 1:
            base += 12
            
        score = base + np.random.normal(0, 6)
        coding[idx] = score
        
    df['Coding Score'] = np.clip(coding, 25.0, 100.0).round(1)
    
    # 8. Technical Interview Score (30 - 100)
    tech_interview = np.zeros(n_rows)
    for idx, row in df.iterrows():
        is_placed = 1 if row['PlacementStatus'] == 'Placed' else 0
        cgpa_val = row['CGPA']
        skills_val = row['Skills']
        
        base = 40 + (cgpa_val - 6.5) * 8 + (skills_val - 5) * 5
        if is_placed == 1:
            base += 15
            
        score = base + np.random.normal(0, 5)
        tech_interview[idx] = score
        
    df['Technical Interview Score'] = np.clip(tech_interview, 30.0, 100.0).round(1)
    
    # 9. Programming Languages Known
    languages_pool = ["Python", "Java", "C++", "JavaScript", "SQL", "Go", "Kotlin", "Swift"]
    langs_known = []
    for idx, row in df.iterrows():
        dept = row['Department']
        skills_val = row['Skills'] # rating 1-10
        
        num_langs = 1
        if skills_val >= 8:
            num_langs = np.random.randint(3, 6)
        elif skills_val >= 6:
            num_langs = np.random.randint(2, 4)
        else:
            num_langs = np.random.randint(1, 3)
            
        # Prioritize languages based on department
        if dept in ["Computer Science & Engineering", "Information Technology"]:
            p = [0.25, 0.20, 0.15, 0.20, 0.15, 0.02, 0.01, 0.02]
        elif dept == "Electronics & Communication Engineering":
            p = [0.20, 0.15, 0.25, 0.10, 0.15, 0.05, 0.05, 0.05]
        else:
            p = [0.30, 0.10, 0.15, 0.10, 0.20, 0.05, 0.05, 0.05]
            
        # Normalize probabilities
        p = np.array(p) / sum(p)
        chosen = np.random.choice(languages_pool, size=min(num_langs, len(languages_pool)), replace=False, p=p)
        langs_known.append(", ".join(chosen))
        
    df['Programming Languages Known'] = langs_known
    
    # 10. Soft Skills Rating (1 - 5)
    # Map from original Communication Skill Rating (which is 1 to 4.8)
    soft_skills = np.zeros(n_rows)
    comm_rating = df['Communication Skill Rating'].values
    for idx, rating in enumerate(comm_rating):
        # Scale rating from [1.0, 4.8] to [1.0, 5.0] and add noise
        val = (rating - 1) / (4.8 - 1) * 4 + 1
        soft_skills[idx] = np.clip(val + np.random.normal(0, 0.2), 1.0, 5.0)
    df['Soft Skills Rating'] = soft_skills.round(1)
    
    # Generate 0-100 Communication Score based on original rating
    comm_scores = np.zeros(n_rows)
    for idx, rating in enumerate(comm_rating):
        # Map rating [1.0, 4.8] to [30.0, 100.0] scale with noise
        val = (rating - 1.0) / (4.8 - 1.0) * 66.0 + 30.0
        comm_scores[idx] = np.clip(val + np.random.normal(0, 3.5), 30.0, 100.0)
    df['Communication Score'] = comm_scores.round(1)
    
    # 11. Extra Curricular Activities (Yes/No)
    df['Extra Curricular Activities'] = np.random.choice(["Yes", "No"], size=n_rows, p=[0.4, 0.6])
    
    # 12. Expected Salary (LPA) - Target for Regression
    # Placed students: 3.5 to 45.0 LPA. Unplaced: 0.0, or potential: 2.5 to 5.5 LPA.
    # Let's make it represent student's salary valuation (Expected Salary). Placed have higher, Unplaced have lower.
    salaries = np.zeros(n_rows)
    for idx, row in df.iterrows():
        is_placed = 1 if row['PlacementStatus'] == 'Placed' else 0
        cgpa_val = row['CGPA']
        coding_val = df['Coding Score'].values[idx]
        intern_val = 1 if row['Internship'] == 'Yes' else 0
        hack_val = 1 if row['Hackathon'] == 'Yes' else 0
        cert_val = row['Workshops/Certificatios']
        proj_val = row['Major Projects'] + row['Mini Projects']
        
        if is_placed == 1:
            # Base salary is 3.5 LPA
            base = 3.5
            # CGPA effect
            base += (cgpa_val - 6.5) * 3.5
            # Coding effect
            base += (coding_val - 30) * 0.15
            # Projects, Internships, Hackathons
            base += intern_val * 2.5
            base += hack_val * 1.5
            base += proj_val * 0.8
            base += cert_val * 0.5
            
            # Super dream packages for top-tier students
            if cgpa_val >= 8.5 and coding_val >= 85 and intern_val == 1:
                base += np.random.uniform(10, 20)
                
            salary = base + np.random.normal(0, 1.2)
            salaries[idx] = np.clip(salary, 3.5, 45.0)
        else:
            # Unplaced potential salary (market value if they get placed after training)
            base = 2.5 + (cgpa_val - 6.5) * 1.2 + (coding_val - 30) * 0.05 + intern_val * 0.8 + cert_val * 0.3
            salary = base + np.random.normal(0, 0.4)
            salaries[idx] = np.clip(salary, 2.5, 5.8)
            
    df['Expected Salary'] = salaries.round(2)
    
    # 13. Map original columns to user friendly names or keep them consistent
    # Keep original columns, but rename some for clarity
    df.rename(columns={
        'Workshops/Certificatios': 'Certifications',
        'Skills': 'Technical Skills Rating'
    }, inplace=True)
    df.drop(['Communication Skill Rating'], axis=1, errors='ignore', inplace=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved enriched dataset to: {output_file} with shape {df.shape}")

if __name__ == "__main__":
    enrich_dataset()
