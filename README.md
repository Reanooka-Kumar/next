# 🌌 NextCareer AI

NextCareer AI is a professional, production-quality AI-powered career intelligence and placement accelerator suite. It evaluates student academic and professional profiles to predict placement success, identify strategic upskilling needs, score resume ATS suitability, and host interactive live mock interview simulations designed to secure target offers.

---

## 🚀 Key Features

* **🛡️ Secure Access Gateway**: Translucent glassmorphic login card overlayed on a clean, high-contrast matte dark grid layout (`#0B0F19`) with radial indigo highlights and built-in role-based access control (RBAC).
  * **Student View**: Restricts access to individual diagnostic panels (*My Prediction*, *My Skill Gap*, *My Resume ATS Scorer*, and *AI Mock Interview*) to safeguard privacy.
  * **Admin View**: Displays the complete institutional analytics suite, model training, and database configurations.
* **📊 Institutional Analytics Dashboard**: Features high-fidelity KPIs (Total Students, Placement Rate, Avg CGPA, Coding Score), department-wise bar charts, academic-vs-coding scatter plots, and correlation heatmaps.
* **🔮 ML Classification & Regression Engines**:
  * **Placement Predictor**: Preprocesses, encodes, and scales student inputs to output a binary placement forecast, placement probability percentage, and forecast confidence classification.
  * **Salary Package Regressor**: Employs a Random Forest Regressor to compute expected salary packages in LPA with a 95% Confidence Interval.
* **🔍 Explainable AI (XAI)**: Visualizes feature contributions using custom Plotly waterfall and horizontal contribution charts, indicating exact positive and negative percentage impacts.
* **🤖 AI Mock Interview Coach**: Generates role-specific behavioral and technical interview questions dynamically, evaluating responses semantically with detailed feedback and scoring powered by the Google Gemini API.
* **🎯 Advisory & Career Roadmap Modules**:
  * **Career Matchmaker**: Scores compatibility percentages across 7 major job roles with targeted advice.
  * **Company Fit Analysis**: Evaluates matching probability across 11 tech employers grouped by Tier 1, Tier 2, and Tier 3.
  * **Skill Gap Diagnostic & 12-Week Roadmap Checklist**: Identifies missing competencies, suggests certifications, estimates potential probability gains, and creates a tailored study plan.
* **📝 Resume ATS Analyzer**: Parses PDF resumes using `pypdf`, checks for core sections, scores keyword matches, and provides recommendations for improvements.
* **💾 Data Management System**: Connects to a local SQLite database log to search query history, delete logs, or export diagnostic predictions to CSV.

---

## 🔑 Demo Credentials

| Role | Username | Password |
| :--- | :--- | :--- |
| **Administrator** | `admin` | `admin123` |
| **Student** | `student` | `student123` |

---

## 🛠️ Installation & Setup

### Prerequisites
* **Python 3.12** or higher
* Git
* Google AI Studio Gemini API Key (saved securely in `.env`)

### Step-by-Step Configuration

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Reanooka-Kumar/student_placement_prediction.git
   cd student_placement_prediction
   ```

2. **Configure API Credentials**:
   Create a `.env` file in the root directory:
   ```bash
   GEMINI_API_KEY=your_google_ai_studio_api_key_here
   ```

3. **Set up a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Generate Enriched Data & Train Models**:
   ```bash
   # Generate synthetic data with aligned scales
   python generate_data.py
   
   # Train classification and regression models
   python src/model_training.py
   ```

6. **Run the Streamlit Dashboard**:
   ```bash
   python -m streamlit run app.py
   ```

---

## 🧪 Running Automated Tests

Validate all modular components (Predictor, Explainer, ATS Scorer, AI Coach, Advisory, PDF Generator) by running the test suite:
```bash
python test_platform.py
```

---

## 📂 Project Structure

```
├── app.py                       # Main Streamlit UI Entrypoint
├── generate_data.py             # Enriched Dataset Generator
├── test_platform.py             # Automated unit and integration test suite
├── requirements.txt             # Python packages listing
├── .gitignore                   # Files excluded from git tracking
├── .env                         # Local environment configuration (API keys, ignored in git)
├── models/                      # Saved trained models, scalers, and metric JSONs
│   ├── best_classifier.joblib
│   ├── salary_regressor.joblib
│   └── scaler.joblib
├── src/                         # Backend Logical Modules
│   ├── data_manager.py          # SQLite prediction log transactions
│   ├── data_preprocessing.py    # Standardizing, scaling, and dummies mapping
│   ├── prediction.py            # Classification & Regression inference engine
│   ├── explainability.py        # SHAP calculation & contribution visuals
│   ├── interview_coach.py       # Gemini-based generative mock interview grader
│   ├── resume_analyzer.py       # PDF ATS scoring parser
│   ├── report_generator.py      # ReportLab PDF compiler
│   ├── career_recommendation.py # Job roles compatibility scorecards
│   ├── company_recommendation.py# Employer tiers matching
│   ├── skill_gap.py             # Missing skills analysis
│   └── roadmap.py               # Personalized 12-week preparation study plan
└── README.md                    # Technical documentation (this file)
```
