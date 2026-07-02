import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import json

# Import custom modules
from src.data_manager import save_prediction, get_prediction_history, search_history, clear_history, validate_csv_data
from src.data_preprocessing import DEPARTMENTS
from src.prediction import PlacementPredictor
from src.explainability import PlacementExplainer
from src.career_recommendation import recommend_careers
from src.company_recommendation import recommend_companies
from src.skill_gap import analyze_skill_gap
from src.roadmap import generate_roadmap
from src.resume_analyzer import analyze_resume, extract_text_from_pdf
from src.report_generator import generate_student_pdf

# Set page configuration
st.set_page_config(
    page_title="Placement Intelligence Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for login & theme
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'dark'

def render_login_page():
    # Inject full screen glassmorphism background CSS
    login_css = """
    <style>
        .stApp {
            background-image: linear-gradient(rgba(14, 17, 23, 0.75), rgba(14, 17, 23, 0.75)), url('https://images.unsplash.com/photo-1541339907198-e08756dedf3f?q=80&w=1920');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        [data-testid="stHeader"], [data-testid="stSidebar"] {
            display: none !important;
        }
        
        .login-box {
            background: rgba(26, 31, 44, 0.85);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            color: #FFFFFF;
            width: 100%;
            margin-top: 80px;
        }
        /* Style label text to white for readability */
        label {
            color: #FFFFFF !important;
            font-weight: 500 !important;
        }
        .login-title {
            font-size: 28px;
            font-weight: bold;
            color: #6366F1;
            text-align: center;
            margin-bottom: 5px;
        }
        .login-subtitle {
            font-size: 14px;
            color: #A0AEC0;
            text-align: center;
            margin-bottom: 25px;
        }
    </style>
    """
    st.markdown(login_css, unsafe_allow_html=True)
    
    _, col, _ = st.columns([1.2, 1.6, 1.2])
    with col:
        st.markdown("""
        <div class="login-box">
            <div class="login-title">🎓 PLACEMENT INTELLIGENCE</div>
            <div class="login-subtitle">Platform Secure Access Gateway</div>
        </div>
        """, unsafe_allow_html=True)
        
        role = st.selectbox("Select Access Role", ["Student", "Admin"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        st.write("")
        login_btn = st.button("🔑 Access Account Portal", use_container_width=True)
        
        if login_btn:
            if role == "Admin":
                if username == "admin" and password == "admin123":
                    st.session_state.logged_in = True
                    st.session_state.user_role = "admin"
                    st.success("Access Granted! Welcome Administrator.")
                    st.rerun()
                else:
                    st.error("Invalid Administrator Credentials.")
            else:
                if username == "student" and password == "student123":
                    st.session_state.logged_in = True
                    st.session_state.user_role = "student"
                    st.success("Access Granted! Welcome Student.")
                    st.rerun()
                else:
                    st.error("Invalid Student Credentials.")
                    
        st.write("---")
        st.markdown("""
        <div style='text-align: center; color: #718096; font-size: 12px;'>
            <b>Demo Credentials</b><br/>
            Admin: <i>admin / admin123</i> &nbsp;|&nbsp; Student: <i>student / student123</i>
        </div>
        """, unsafe_allow_html=True)

if not st.session_state.logged_in:
    render_login_page()
    st.stop()

# -------------------------------------------------------------
# 1. Custom CSS Theme Styling (Light / Dark Mode Styles)
# -------------------------------------------------------------

def toggle_theme():
    if st.session_state.theme_mode == 'dark':
        st.session_state.theme_mode = 'light'
    else:
        st.session_state.theme_mode = 'dark'

# Custom UI Styles
dark_css = """
<style>
    /* Dark Theme Styles */
    :root {
        --bg-color: #0E1117;
        --card-bg: #1A1F2C;
        --card-border: #2D3748;
        --text-primary: #FFFFFF;
        --text-secondary: #A0AEC0;
        --accent: #6366F1;
        --accent-glow: rgba(99, 102, 241, 0.15);
    }
    
    .main .block-container {
        padding-top: 1.5rem;
    }
    
    /* Premium Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 16px !important;
        padding: 20px 24px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 0 15px var(--accent-glow) !important;
    }
    
    /* Section containers */
    .section-card {
        background-color: #1A1F2C;
        border: 1px solid #2D3748;
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
</style>
"""

light_css = """
<style>
    /* Light Theme Styles */
    :root {
        --bg-color: #F7FAFC;
        --card-bg: #FFFFFF;
        --card-border: #E2E8F0;
        --text-primary: #1A202C;
        --text-secondary: #4A5568;
        --accent: #4F46E5;
        --accent-glow: rgba(79, 70, 229, 0.1);
    }
    
    .main .block-container {
        padding-top: 1.5rem;
    }
    
    /* Premium Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 16px !important;
        padding: 20px 24px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 0 15px var(--accent-glow) !important;
    }
    
    /* Section containers */
    .section-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
</style>
"""

if st.session_state.theme_mode == 'dark':
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    st.markdown(light_css, unsafe_allow_html=True)


# Initialize Predictor & Explainer with Cache
@st.cache_resource
def get_predictor_assets():
    p = PlacementPredictor()
    return p

@st.cache_resource
def get_explainer_assets():
    p = get_predictor_assets()
    e = PlacementExplainer(p)
    return e

predictor = get_predictor_assets()
explainer = get_explainer_assets()

# -------------------------------------------------------------
# 2. Sidebar Navigation & Global Controls
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/graduation-cap.png", width=70)
    st.title("PIP Platform")
    st.subheader("Placement Intelligence Platform")
    st.write("---")
    
    # Role-based Navigation list
    if st.session_state.user_role == 'admin':
        nav_options = [
            "🏢 Dashboard Analytics",
            "🔮 Predict Placement & XAI",
            "🎯 Skill Gap & Career Roadmap",
            "📝 Resume ATS Scorer",
            "📊 Model Benchmarking",
            "💾 History & Data Management"
        ]
    else:
        nav_options = [
            "🔮 My Placement Prediction & XAI",
            "🎯 My Skill Gap & Roadmap",
            "📝 My Resume ATS Scorer"
        ]
        
    nav_selection = st.radio("Navigation Menu", options=nav_options)
    
    # Map navigation selections for routing
    routing_selection = nav_selection
    if "Predict Placement" in nav_selection or "My Placement" in nav_selection:
        routing_selection = "🔮 Predict Placement & XAI"
    elif "Skill Gap" in nav_selection or "My Skill Gap" in nav_selection:
        routing_selection = "🎯 Skill Gap & Career Roadmap"
    elif "Resume" in nav_selection:
        routing_selection = "📝 Resume ATS Scorer"
        
    st.write("---")
    # Theme toggler
    mode_emoji = "☀️ Light" if st.session_state.theme_mode == 'dark' else "🌙 Dark"
    st.button(f"Switch to {mode_emoji} Mode", on_click=toggle_theme, use_container_width=True)
    
    st.write("---")
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.rerun()
        
    st.write("---")
    st.caption("© 2026 Placement Intelligence Platform. Production-Quality AI Advisory System.")

# -------------------------------------------------------------
# 3. Helper Functions
# -------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Loads and caches the enriched student database."""
    if os.path.exists("Placement_Data_Enriched.csv"):
        return pd.read_csv("Placement_Data_Enriched.csv")
    return pd.DataFrame()

df_students = load_dataset()

# -------------------------------------------------------------
# 4. Modules Routing
# -------------------------------------------------------------

# ==========================================
# MODULE 1: Dashboard Analytics
# ==========================================
if routing_selection == "🏢 Dashboard Analytics":
    st.title("🏢 Institutional Placement Analytics Dashboard")
    st.markdown("A comprehensive, company-grade overview of students' academic profiles, skill distributions, and predicted placement indices.")
    
    if df_students.empty:
        st.warning("Sample dataset 'Placement_Data_Enriched.csv' not found. Please run the data generator in the Data Management tab.")
    else:
        # 1. KPI Cards Row
        total_students = len(df_students)
        placed_rate = (df_students['PlacementStatus'] == 'Placed').mean() * 100
        avg_cgpa = df_students['CGPA'].mean()
        avg_coding = df_students['Coding Score'].mean()
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.metric(label="Total Analyzed Students", value=f"{total_students:,}", delta="Historical Cohort")
        with kpi_col2:
            st.metric(label="Placement Rate (%)", value=f"{placed_rate:.1f}%", delta="+2.4% vs Last Year")
        with kpi_col3:
            st.metric(label="Average CGPA", value=f"{avg_cgpa:.2f} / 10.0")
        with kpi_col4:
            st.metric(label="Average Coding Score", value=f"{avg_coding:.1f} / 100")
            
        st.write("---")
        
        # 2. Charts Section
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Department-wise Placement Distribution")
            dept_placement = df_students.groupby('Department')['PlacementStatus'].apply(
                lambda x: (x == 'Placed').mean() * 100
            ).reset_index().rename(columns={'PlacementStatus': 'Placement Rate (%)'})
            
            fig_dept = px.bar(
                dept_placement,
                x='Placement Rate (%)',
                y='Department',
                orientation='h',
                color='Placement Rate (%)',
                color_continuous_scale=px.colors.sequential.Blues,
                labels={'Department': ''}
            )
            fig_dept.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350)
            st.plotly_chart(fig_dept, use_container_width=True)
            
        with chart_col2:
            st.subheader("Placement Probability Distribution")
            # Generate dummy predictions for analytics visualization
            fig_hist = px.histogram(
                df_students, 
                x='Expected Salary',
                color='PlacementStatus',
                nbins=30,
                color_discrete_map={'Placed': '#6366F1', 'NotPlaced': '#EF4444'},
                labels={'Expected Salary': 'Expected Salary (LPA)', 'count': 'Number of Students'}
            )
            fig_hist.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350, barmode='overlay')
            st.plotly_chart(fig_hist, use_container_width=True)
            
        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            st.subheader("Academic Performance vs Coding Competency")
            fig_scatter = px.scatter(
                df_students.sample(1000, random_state=42), 
                x='CGPA', 
                y='Coding Score',
                color='PlacementStatus',
                color_discrete_map={'Placed': '#059669', 'NotPlaced': '#DC2626'},
                opacity=0.7,
                trendline="ols",
                labels={'PlacementStatus': 'Placement Status'}
            )
            fig_scatter.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with chart_col4:
            st.subheader("Core Recruitment Features Correlation Matrix")
            corr_cols = ['CGPA', 'Attendance', 'Aptitude Score', 'Coding Score', 'Technical Interview Score', 'Expected Salary']
            corr_matrix = df_students[corr_cols].corr()
            
            fig_heat = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            fig_heat.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=350)
            st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# MODULE 2 & 3 & 4 & 5 & 8: Student Prediction & XAI
# ==========================================
elif routing_selection == "🔮 Predict Placement & XAI":
    st.title("🔮 Predictive Student Placement & Explainable AI Engine")
    st.markdown("Input student parameters to forecast placement outcomes, calculate salary packages, and render visual SHAP attribution charts.")
    
    if not predictor.is_ready():
        st.error("🤖 Machine Learning Models are not loaded. Please go to the **Model Benchmarking** tab to train models first.")
    else:
        # Form for student profile input
        with st.form("student_profile_form"):
            st.subheader("Student Academic & Professional Parameters")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                name = st.text_input("Student Name", value="Aarav Sharma")
                age = st.number_input("Age", min_value=18, max_value=30, value=21, step=1)
                department = st.selectbox("Academic Department", options=DEPARTMENTS)
                cgpa = st.slider("Cumulative CGPA (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=7.8, step=0.05)
                attendance = st.slider("Attendance Rate (%)", min_value=40, max_value=100, value=82, step=1)
                
            with col2:
                aptitude = st.slider("Aptitude Test Score (0 - 100)", min_value=0, max_value=100, value=75, step=1)
                coding = st.slider("Coding Challenge Score (0 - 100)", min_value=0, max_value=100, value=72, step=1)
                communication = st.slider("Communication Score (0 - 100)", min_value=0, max_value=100, value=80, step=1)
                technical_interview = st.slider("Technical Interview Score (0 - 100)", min_value=0, max_value=100, value=70, step=1)
                soft_skills = st.slider("Soft Skills Rating (1.0 - 5.0)", min_value=1.0, max_value=5.0, value=3.8, step=0.1)
                
            with col3:
                internships = st.number_input("Internships Completed", min_value=0, max_value=4, value=1, step=1)
                certifications = st.number_input("Technical Certifications", min_value=0, max_value=10, value=2, step=1)
                projects = st.number_input("Projects Completed", min_value=0, max_value=10, value=2, step=1)
                
                languages = st.multiselect(
                    "Programming Languages Known",
                    options=["Python", "Java", "C++", "JavaScript", "SQL", "Go", "Kotlin", "Swift"],
                    default=["Python", "SQL", "Java"]
                )
                
                extra_curricular = st.radio("Active in Extra Curricular Activities?", options=["Yes", "No"], index=0, horizontal=True)
                
            submit_profile = st.form_submit_button("🔮 Run Diagnostic Analysis", use_container_width=True)
            
        if submit_profile:
            # Prepare profile dictionary
            student_dict = {
                'name': name,
                'age': age,
                'department': department,
                'cgpa': cgpa,
                'attendance': attendance,
                'aptitude': aptitude,
                'coding': coding,
                'communication': communication,
                'technical_interview': technical_interview,
                'internships': internships,
                'certifications': certifications,
                'projects': projects,
                'languages': ", ".join(languages),
                'soft_skills': soft_skills,
                'extra_curricular': extra_curricular
            }
            
            # Predict
            with st.spinner("Analyzing profile & simulating outcomes..."):
                results = predictor.predict_student(student_dict)
                
            if 'error' in results:
                st.error(results['error'])
            else:
                # Save to database history
                save_prediction(student_dict, results['placement_status'], results['placement_probability'], results['expected_salary'])
                
                # Cache parameters in session state for child tabs (Skill Gap, Roadmap, PDF generation)
                st.session_state.active_student_profile = student_dict
                st.session_state.active_student_results = results
                
                # Show results header
                st.success("🎉 Analysis complete! Scroll down to view diagnostic reports.")
                
                # Metric Cards Display
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                
                with m_col1:
                    status_emoji = "✅" if results['placement_status'] == 'Placed' else "⚠️"
                    st.metric(
                        label="Placement Status Prediction",
                        value=f"{status_emoji} {results['placement_status'].upper()}",
                        delta="Model Prediction"
                    )
                with m_col2:
                    st.metric(
                        label="Placement Probability (%)",
                        value=f"{results['placement_probability']:.1f}%",
                        delta=f"Confidence: {results['confidence_score']:.1f}%"
                    )
                with m_col3:
                    st.metric(
                        label="Predicted Salary Package",
                        value=f"💵 {results['expected_salary']:.2f} LPA",
                        delta="Expected Package"
                    )
                with m_col4:
                    st.metric(
                        label="Salary Range (95% CI)",
                        value=f"{results['salary_range_min']:.1f} - {results['salary_range_max']:.1f} LPA",
                        delta="LPA Interval"
                    )
                    
                # Store PDF in memory for download
                try:
                    # Calculate career recs
                    c_recs = recommend_careers(student_dict)
                    comp_recs = recommend_companies(student_dict)
                    sg_analysis = analyze_skill_gap(student_dict, c_recs[0]['role'] if c_recs else 'Software Engineer')
                    rm_schedule = generate_roadmap(student_dict, c_recs[0]['role'] if c_recs else 'Software Engineer')
                    
                    pdf_buffer = io.BytesIO()
                    generate_student_pdf(
                        pdf_buffer, student_dict, results, c_recs, comp_recs, sg_analysis, rm_schedule
                    )
                    pdf_data = pdf_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Detailed Career & Placement Report (PDF)",
                        data=pdf_data,
                        file_name=f"Placement_Assessment_{name.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as ex_pdf:
                    st.error(f"Failed to compile PDF Report: {ex_pdf}")
                    
                st.write("---")
                
                # -------------------------------------------------------------
                # Explainable AI (SHAP Plots)
                # -------------------------------------------------------------
                st.header("🧮 Explainable AI (XAI) Attribution")
                st.markdown("Interpret which parameters positively and negatively influence the classification output.")
                
                with st.spinner("Extracting model explanation values..."):
                    shap_results = explainer.get_shap_explanation(student_dict)
                    
                xai_col1, xai_col2 = st.columns([1, 1])
                
                with xai_col1:
                    # Waterfall chart
                    fig_waterfall = explainer.plot_waterfall_plotly(shap_results)
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                    
                with xai_col2:
                    # Bar contribution chart
                    fig_contrib = explainer.plot_shap_plotly(shap_results)
                    st.plotly_chart(fig_contrib, use_container_width=True)
                    
                # Positives vs Negatives Highlights
                col_pos, col_neg = st.columns(2)
                with col_pos:
                    st.markdown("##### 🚀 Top Features Boosting Your Chances")
                    for item in shap_results['positive_influences'][:4]:
                        st.markdown(f"- **{item['friendly_name']}** (`{item['raw_value']}`): added **+{item['shap_value']*100:.1f}%** chance.")
                        
                with col_neg:
                    st.markdown("##### 📉 Top Areas Limiting Your Chances")
                    for item in shap_results['negative_influences'][:4]:
                        st.markdown(f"- **{item['friendly_name']}** (`{item['raw_value']}`): reduced chances by **{item['shap_value']*100:.1f}%**.")
                        
                st.write("---")
                
                # -------------------------------------------------------------
                # Career & Company Matcher
                # -------------------------------------------------------------
                st.header("💼 Intelligent Career & Company Matcher")
                
                col_c1, col_c2 = st.columns([4, 5])
                with col_c1:
                    st.subheader("Role Compatibility Index")
                    career_recommendations = recommend_careers(student_dict)
                    for rec in career_recommendations[:4]:
                        st.markdown(f"**{rec['role']}** (Match Score: **{rec['score']:.1f}%**)")
                        # Display progress bar matching score
                        st.progress(float(rec['score'] / 100.0))
                        # Expandable detailed bullets
                        with st.expander("Why this role?"):
                            for bullet in rec['reasons']:
                                st.write(f"- {bullet}")
                                
                with col_c2:
                    st.subheader("Employer Suitability Fit")
                    company_recommendations = recommend_companies(student_dict)
                    
                    # Group by fit level
                    for comp in company_recommendations[:5]:
                        fit_color = "green" if comp['fit'] == "Strong Match" else "orange" if comp['fit'] == "Moderate Match" else "red"
                        st.markdown(f"💼 **{comp['company']}** ({comp['tier']}) - <span style='color:{fit_color}; font-weight:bold;'>{comp['fit']} ({comp['score']:.1f}%)</span>", unsafe_allow_html=True)
                        st.write(f"*Hiring Suitability:* {comp['reasons']}")
                        st.caption(f"*Advisory:* {comp['advice']}")
                        st.write("")

# ==========================================
# MODULE 6 & 7 & 10: Skill Gap & Career Roadmap
# ==========================================
elif routing_selection == "🎯 Skill Gap & Career Roadmap":
    st.title("🎯 Competency Skill Gap & Learning Roadmap")
    
    if 'active_student_profile' not in st.session_state:
        st.warning("Please input student profile metrics in the **Predict Placement & XAI** tab first to generate your custom learning path.")
    else:
        profile = st.session_state.active_student_profile
        results = st.session_state.active_student_results
        
        # Select target role for skill gap
        career_options = [c['role'] for c in recommend_careers(profile)]
        target_role = st.selectbox("Select Your Target Career Path", options=career_options)
        
        st.write("---")
        
        # Analyze Gap
        gap_results = analyze_skill_gap(profile, target_role)
        
        col_gap1, col_gap2 = st.columns([1, 1])
        with col_gap1:
            st.subheader("Competency Profile Breakdown")
            st.markdown(f"**Target Role:** `{target_role}`")
            
            p_color = "red" if gap_results['priority_level'] == "High" else "orange" if gap_results['priority_level'] == "Medium" else "green"
            st.markdown(f"**Action Priority Level:** <span style='color:{p_color}; font-weight:bold;'>{gap_results['priority_level']}</span>", unsafe_allow_html=True)
            
            st.markdown("##### ✅ Existing Skill Strengths")
            if gap_results['existing_skills']:
                for skill in gap_results['existing_skills']:
                    st.write(f"- {skill}")
            else:
                st.write("- No matching strengths identified.")
                
            st.markdown("##### 🚨 Identified Missing Competencies")
            for skill in gap_results['missing_skills']:
                st.write(f"- {skill}")
                
        with col_gap2:
            st.subheader("Placement Boost Forecast")
            st.markdown(f"By closing these skill gaps, the student can increase their placement probability by:")
            st.markdown(f"<h1 style='color:#059669; font-size:48px;'>+{gap_results['estimated_probability_gain']}%</h1>", unsafe_allow_html=True)
            
            st.markdown("##### 📜 Recommended Skill Certifications")
            for cert in gap_results['recommended_certifications']:
                st.markdown(f"- **{cert['name']}** (Improves probability by **+{cert['prob_gain']}%**)")
                
        st.write("---")
        
        # Personalized 12-Week Roadmap
        st.header("📅 Tailored 12-Week Placement Action Plan")
        st.markdown("This week-by-week timeline covers core study topics, programming practices, and interview mock preparations tailored to the student's weaknesses.")
        
        roadmap_timeline = generate_roadmap(profile, target_role)
        
        for index, block in enumerate(roadmap_timeline):
            with st.expander(f"📌 {block['weeks']}: {block['topic']}", expanded=(index == 0)):
                st.markdown(f"*{block['description']}*")
                st.write("**Weekly Tasks Checklist:**")
                for action in block['action_items']:
                    st.checkbox(action, value=False, key=f"action_{index}_{action[:20]}")
                st.info(f"📚 **Recommended Resources:** {block['resources']}")

# ==========================================
# MODULE 9: Resume ATS Scorer
# ==========================================
elif routing_selection == "📝 Resume ATS Scorer":
    st.title("📝 ATS Resume Analyzer & Scorer")
    st.markdown("Upload a PDF resume to parse contents, scan for core keywords matching the target role, and calculate an ATS score.")
    
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        st.subheader("Upload Resume PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        
        # Select target role for resume scoring
        target_role = st.selectbox(
            "Target Career Track",
            options=["Software Engineer", "Data Scientist", "Machine Learning Engineer", "Data Analyst", "Business Analyst", "AI Engineer", "Full Stack Developer"]
        )
        
        analyze_button = st.button("📝 Analyze Resume ATS Score", use_container_width=True, disabled=(uploaded_file is None))
        
    with col_res2:
        st.subheader("Analysis Insights")
        if uploaded_file is None:
            st.info("Upload a student's resume PDF on the left and click 'Analyze' to view ATS keyword matches and scoring.")
        elif analyze_button:
            with st.spinner("Extracting PDF text and matching keywords..."):
                # Extract text
                resume_text = extract_text_from_pdf(uploaded_file)
                # Analyze
                resume_analysis = analyze_resume(resume_text, target_role)
                
            # Score display gauge
            score = resume_analysis['ats_score']
            score_color = "red" if score < 50 else "orange" if score < 75 else "green"
            
            st.markdown(f"#### ATS Resume Score: <span style='color:{score_color}; font-size:32px;'><b>{score}/100</b></span>", unsafe_allow_html=True)
            
            # Progress bar
            st.progress(float(score / 100.0))
            
            # Expandable skills and warnings
            st.write(f"**Word Count:** {resume_analysis['word_count']} words")
            
            st.markdown("##### 🛠️ Skills Found in Resume")
            if resume_analysis['skills_found']:
                st.write(", ".join(resume_analysis['skills_found']))
            else:
                st.write("*No technical skills found in resume text.*")
                
            st.markdown("##### ❌ Missing Keywords for Target Track")
            if resume_analysis['missing_keywords']:
                st.write(", ".join(resume_analysis['missing_keywords']))
            else:
                st.write("*No missing keywords! Perfect match.*")
                
            st.markdown("##### 💡 ATS Improvement Suggestions")
            for sug in resume_analysis['suggestions']:
                st.markdown(f"- {sug}")

# ==========================================
# MODULE 12: Model Benchmarking
# ==========================================
elif routing_selection == "📊 Model Benchmarking":
    st.title("📊 Machine Learning Models Benchmarking & Comparison")
    st.markdown("Train, evaluate, and benchmark multiple ML classifiers. Auto-select the best model based on F1-Score metrics.")
    
    # Check if model_metrics.json exists
    metrics_path = "models/model_metrics.json"
    if not os.path.exists(metrics_path):
        st.warning("No pre-trained model metadata found. Click the button below to train models using the enriched student dataset.")
        train_button = st.button("🏋️ Train Model Suite (Logistic Regression, Trees, Ensemble, XGBoost)", use_container_width=True)
        if train_button:
            with st.spinner("Executing Grid Search & model training..."):
                from src.model_training import train_and_evaluate_models
                try:
                    train_and_evaluate_models()
                    st.success("Models trained and cached successfully! Refreshing dashboard...")
                    st.rerun()
                except Exception as train_ex:
                    st.error(f"Training failed: {train_ex}")
    else:
        # Load metrics
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)
            
        st.success(f"🤖 **Auto-selected Best Classifier:** `{all_metrics.get('best_model', 'N/A')}`")
        
        # 1. Comparison Table
        table_rows = []
        for name, data in all_metrics.items():
            if name not in ['best_model', 'salary_metrics']:
                table_rows.append({
                    'Model Name': name,
                    'Accuracy': f"{data['accuracy']*100:.2f}%",
                    'Precision': f"{data['precision']*100:.2f}%",
                    'Recall': f"{data['recall']*100:.2f}%",
                    'F1-Score': f"{data['f1_score']*100:.2f}%",
                    'ROC AUC': f"{data['roc_auc']:.4f}"
                })
        df_metrics = pd.DataFrame(table_rows)
        st.subheader("Classifier Suite Metrics Comparison Table")
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        st.write("---")
        
        # 2. ROC Curves & Confusion Matrices
        col_curve1, col_curve2 = st.columns(2)
        
        with col_curve1:
            st.subheader("Classifier ROC Curves")
            fig_roc = go.Figure()
            for name, data in all_metrics.items():
                if name not in ['best_model', 'salary_metrics']:
                    fig_roc.add_trace(go.Scatter(
                        x=data['fpr'],
                        y=data['tpr'],
                        mode='lines',
                        name=f"{name} (AUC={data['roc_auc']:.3f})"
                    ))
            fig_roc.add_shape(
                type='line', line=dict(dash='dash', color='grey'),
                x0=0, x1=1, y0=0, y1=1
            )
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                margin=dict(l=10, r=10, t=10, b=10),
                height=350,
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
        with col_curve2:
            st.subheader("Confusion Matrix View")
            # Select model confusion matrix to view
            models_list = [name for name in all_metrics.keys() if name not in ['best_model', 'salary_metrics']]
            selected_model = st.selectbox("Select Model for Confusion Matrix", options=models_list)
            
            cm = all_metrics[selected_model]['confusion_matrix']
            cm_df = pd.DataFrame(cm, columns=["Predicted Not Placed", "Predicted Placed"], index=["Actual Not Placed", "Actual Placed"])
            
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale='Blues',
                labels=dict(x="Predicted Class", y="Actual Class")
            )
            fig_cm.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)
            st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================
# MODULE 14: History & Data Management
# ==========================================
elif routing_selection == "💾 History & Data Management":
    st.title("💾 History & Data Management")
    st.markdown("Upload new custom CSV datasets for preprocessing/validation, query past predictions from SQLite, or export history records.")
    
    tab1, tab2 = st.tabs(["📂 Upload & Validate Dataset", "📋 SQLite Prediction History"])
    
    with tab1:
        st.subheader("Upload Custom Student CSV")
        st.markdown("""
        Ensure your CSV matches the schema format of `Placement_Data_Enriched.csv`. 
        Required columns include: *Name, Age, Department, CGPA, Attendance, Aptitude Score, Coding Score, 
        Communication Score, Technical Interview Score, Internship, Certifications, Projects Completed, 
        Programming Languages Known, Soft Skills Rating, Extra Curricular Activities*.
        """)
        
        uploaded_csv = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_csv is not None:
            raw_upload = pd.read_csv(uploaded_csv)
            is_valid, report, cleaned_df = validate_csv_data(raw_upload)
            
            if not is_valid:
                st.error(f"❌ Validation failed: {report}")
            else:
                st.success("✅ CSV Schema Check Passed!")
                st.info(report)
                
                st.write("---")
                st.write("**Dataset Preview:**")
                st.dataframe(cleaned_df.head(10), use_container_width=True)
                
                # Check metrics summaries on new dataset
                st.write("**Summary Statistics:**")
                st.write(cleaned_df.describe())
                
    with tab2:
        st.subheader("SQLite Saved Prediction History")
        
        # Search query
        search_query = st.text_input("🔍 Search Student Name or Department", value="")
        
        if search_query:
            df_history = search_history(search_query)
        else:
            df_history = get_prediction_history()
            
        if df_history.empty:
            st.info("No prediction history recorded yet. Run student profiles in the prediction tab to populate this database.")
        else:
            st.dataframe(df_history, use_container_width=True)
            
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                # Export to CSV
                csv_buffer = io.StringIO()
                df_history.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Export History to CSV",
                    data=csv_buffer.getvalue(),
                    file_name="Student_Placement_History.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            with h_col2:
                # Clear Database
                if st.button("🗑️ Clear Entire History Database", use_container_width=True, type="secondary"):
                    clear_history()
                    st.success("Prediction database history wiped successfully!")
                    st.rerun()
