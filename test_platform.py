import os
import sys
import io

# Add workspace to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.prediction import PlacementPredictor
from src.explainability import PlacementExplainer
from src.career_recommendation import recommend_careers
from src.company_recommendation import recommend_companies
from src.skill_gap import analyze_skill_gap
from src.roadmap import generate_roadmap
from src.resume_analyzer import analyze_resume
from src.report_generator import generate_student_pdf

def test_pipeline():
    print("[TEST] Starting Automated Test Suite...")
    
    # 1. Verify Model assets exist
    print("Checking model files...")
    models = ["best_classifier.joblib", "salary_regressor.joblib", "scaler.joblib", "feature_names.joblib", "model_metrics.json"]
    for m in models:
        path = os.path.join("models", m)
        assert os.path.exists(path), f"Missing model asset: {path}"
    print("[PASS] Model files present.")
    
    # 2. Test Predictor
    print("Testing PlacementPredictor loading and inference...")
    predictor = PlacementPredictor()
    assert predictor.is_ready(), "Predictor failed to load assets."
    
    test_student = {
        'name': 'Test Student',
        'age': 22,
        'department': 'Computer Science & Engineering',
        'cgpa': 8.8,
        'attendance': 85.0,
        'aptitude': 90.0,
        'coding': 95.0,
        'communication': 88.0,
        'technical_interview': 92.0,
        'internships': 1,
        'certifications': 3,
        'projects': 3,
        'languages': 'Python, SQL, Java',
        'soft_skills': 4.5,
        'extra_curricular': 'Yes'
    }
    
    results = predictor.predict_student(test_student)
    assert 'error' not in results, f"Prediction error: {results.get('error')}"
    assert results['placement_status'] in ['Placed', 'Not Placed'], "Invalid placement status output"
    assert 0.0 <= results['placement_probability'] <= 100.0, "Invalid placement probability range"
    assert results['expected_salary'] > 0.0, "Expected salary should be positive"
    print(f"[PASS] Predictor working. Predicted Status: {results['placement_status']} ({results['placement_probability']:.2f}%), Expected Salary: {results['expected_salary']:.2f} LPA")
    
    # 3. Test Explainer
    print("Testing PlacementExplainer...")
    explainer = PlacementExplainer(predictor)
    shap_results = explainer.get_shap_explanation(test_student)
    assert 'positive_influences' in shap_results, "Explainer output missing positive influences"
    assert 'negative_influences' in shap_results, "Explainer output missing negative influences"
    print(f"[PASS] Explainer working. Positives: {len(shap_results['positive_influences'])}, Negatives: {len(shap_results['negative_influences'])}")
    
    # 4. Test Career & Company matching
    print("Testing Advisory Matches...")
    career_recs = recommend_careers(test_student)
    assert len(career_recs) == 7, f"Expected 7 recommendations, got {len(career_recs)}"
    assert career_recs[0]['score'] > 0.0, "Score should be non-zero"
    
    company_recs = recommend_companies(test_student)
    assert len(company_recs) == 11, f"Expected 11 recommendations, got {len(company_recs)}"
    print("[PASS] Recommendations working.")
    
    # 5. Test Skill Gap & Roadmap
    print("Testing Skill Gap Analysis & Roadmap Timeline...")
    top_role = career_recs[0]['role']
    skill_gap = analyze_skill_gap(test_student, top_role)
    assert len(skill_gap['recommended_certifications']) > 0, "No certs recommended"
    
    roadmap = generate_roadmap(test_student, top_role)
    assert len(roadmap) == 6, f"Expected 6 blocks of roadmap, got {len(roadmap)}"
    print("[PASS] Skill Gap & Roadmap working.")
    
    # 6. Test PDF Exporter
    print("Testing PDF assessment report generation...")
    pdf_buffer = io.BytesIO()
    generate_student_pdf(
        pdf_buffer, test_student, results, career_recs, company_recs, skill_gap, roadmap
    )
    pdf_data = pdf_buffer.getvalue()
    assert len(pdf_data) > 0, "Generated PDF is empty"
    print(f"[PASS] PDF Exporter working. Compiled PDF size: {len(pdf_data)} bytes")
    
    # 7. Test Resume Analyzer
    print("Testing Resume Analyzer mock...")
    resume_text = "Experienced in Python programming, sql database structures. Created a machine learning project."
    resume_analysis = analyze_resume(resume_text, "Data Scientist")
    assert resume_analysis['ats_score'] > 0, "ATS Score should be positive"
    assert 'Python' in resume_analysis['skills_found'], "Skills found missing Python"
    print(f"[PASS] Resume Analyzer working. ATS Score: {resume_analysis['ats_score']}/100")
    
    print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY! The system is highly stable and production-ready.")

if __name__ == "__main__":
    test_pipeline()
