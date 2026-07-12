import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
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
    page_title="NextCareer AI",
    page_icon="🌌",
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
if 'landing_view' not in st.session_state:
    st.session_state.landing_view = 'home'

def render_login_page():
    # Inject full screen glassmorphism background CSS and Landing styling
    login_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {
            background-color: #0B0F19 !important;
            background-image: 
                radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.15) 0%, rgba(11, 15, 25, 0) 60%),
                linear-gradient(0deg, rgba(255, 255, 255, 0.008) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.008) 1px, transparent 1px) !important;
            background-size: 100% 100%, 40px 40px, 40px 40px !important;
            background-position: center top !important;
        }
        
        [data-testid="stHeader"], [data-testid="stSidebar"] {
            display: none !important;
        }
        
        /* Reduce vertical gap and align header navigation elements to the very top */
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        
        /* Brand container styling */
        .brand-hero {
            padding: 10px;
            color: #FFFFFF;
            text-align: center;
        }
        
        .brand-badge {
            background: rgba(99, 102, 241, 0.15);
            color: #818CF8;
            border: 1px solid rgba(99, 102, 241, 0.3);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 25px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-family: 'Inter', sans-serif;
        }
        
        .brand-title {
            font-family: 'Outfit', sans-serif;
            font-size: 56px;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #FFFFFF 30%, #A5B4FC 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .brand-subtitle {
            font-family: 'Outfit', sans-serif;
            font-size: 22px;
            font-weight: 400;
            color: #818CF8;
            margin-bottom: 25px;
        }
        
        .brand-desc {
            font-family: 'Inter', sans-serif;
            font-size: 15.5px;
            color: #94A3B8;
            line-height: 1.6;
            margin-bottom: 35px;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
            text-align: center !important;
        }
        
        /* Features list styling */
        .feature-card {
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
            height: 100%;
        }
        
        .feature-card:hover {
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.5);
            box-shadow: 0 15px 35px rgba(99, 102, 241, 0.15);
        }
        
        .feature-item {
            display: flex;
            align-items: flex-start;
            text-align: left;
            font-family: 'Inter', sans-serif;
        }
        
        .feature-icon {
            font-size: 24px;
            margin-right: 18px;
            background: rgba(255, 255, 255, 0.05);
            padding: 10px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .feature-text {
            color: #E2E8F0;
        }
        
        .feature-text strong {
            color: #FFFFFF;
            font-size: 15px;
            display: block;
            margin-bottom: 2px;
        }
        
        .feature-text p {
            color: #94A3B8;
            font-size: 13.5px;
            margin: 0;
            line-height: 1.4;
        }
        
        /* Stats banner */
        .stats-banner {
            display: flex;
            gap: 30px;
            margin-top: 50px;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding-top: 40px;
            font-family: 'Outfit', sans-serif;
        }
        
        .stat-box {
            flex: 1;
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 2px;
        }
        
        .stat-label {
            font-size: 12px;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Custom styled card override specifically for the login container only */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(input[type="password"]) {
            background: rgba(30, 41, 59, 0.45) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 1px solid rgba(99, 102, 241, 0.25) !important;
            border-radius: 24px !important;
            padding: 30px !important;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 40px rgba(99, 102, 241, 0.1) !important;
        }
        
        /* Input and Label overrides for visibility */
        label, div[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] span, [data-testid="stWidgetLabel"] {
            color: #FFFFFF !important;
            font-weight: 500 !important;
            font-size: 14.5px !important;
            opacity: 1.0 !important;
        }
        
        /* Force dropdown option item texts to be highly visible */
        div[role="listbox"] ul li, div[role="listbox"] ul li *, [data-baseweb="menu"] li {
            color: #111827 !important;
            background-color: #FFFFFF !important;
        }
        div[role="listbox"] ul li:hover, [data-baseweb="menu"] li:hover {
            background-color: #EEF2FF !important;
            color: #4F46E5 !important;
        }
        
        /* Dropdown active selected display text */
        div[data-baseweb="select"] div {
            color: #FFFFFF !important;
        }
        
        input {
            background-color: rgba(15, 23, 42, 0.75) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
        }
        
        input:focus {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
        }
        
        /* Divider line styling */
        hr {
            border-color: rgba(255, 255, 255, 0.1) !important;
        }
    </style>
    """
    st.markdown(login_css, unsafe_allow_html=True)
    
    # 1. Header Navigation Bar
    col_logo, col_action = st.columns([5, 1])
    with col_logo:
        st.markdown("<h3 style='margin: 0; color: #FFFFFF; font-family: \"Outfit\", sans-serif; font-weight: 700; letter-spacing: -0.5px;'>🌌 NextCareer AI</h3>", unsafe_allow_html=True)
    with col_action:
        if st.session_state.landing_view == 'home':
            if st.button("🔒 Sign In", use_container_width=True, type="secondary"):
                st.session_state.landing_view = 'login'
                st.rerun()
        else:
            if st.button("← Back Home", use_container_width=True, type="secondary"):
                st.session_state.landing_view = 'home'
                st.rerun()
                
    st.write("---")
    
    # 2. Main Stateful Router Layout
    if st.session_state.landing_view == 'home':
        # Hero Branding Page
        st.markdown("""
        <div class="brand-hero">
            <div class="brand-badge">🚀 AI-POWERED CAREER INTELLIGENCE</div>
            <div class="brand-title">Predict. Prepare. Elevate.</div>
            <div class="brand-subtitle">Placement Insights & Talent Accelerator Suite</div>
            <p class="brand-desc" style="text-align: center !important;">
                NextCareer AI evaluates your academic and professional profile to predict placement success, 
                identify strategic upskilling needs, and host mock simulations designed to secure your target offer.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Access Portal Center Button
        _, col_center_btn, _ = st.columns([1.3, 1, 1.3])
        with col_center_btn:
            if st.button("⚡ Get Started / Enter Portal", use_container_width=True, type="primary"):
                st.session_state.landing_view = 'login'
                st.rerun()
                
        st.write("")
        st.write("")
        st.write("")
        
        # Core Capabilities Section (4 Cards layout)
        st.markdown("<h3 style='text-align: center; color: #FFFFFF; font-family: \"Outfit\", sans-serif; margin-bottom: 30px;'>Core Capabilities</h3>", unsafe_allow_html=True)
        
        col_f1, col_f2 = st.columns(2, gap="medium")
        with col_f1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-item">
                    <div class="feature-icon">🔮</div>
                    <div class="feature-text">
                        <strong>Placement Predictor</strong>
                        <p>Project your career outcomes and initial packages based on academic benchmarks.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
                
        with col_f2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-item">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-text">
                        <strong>Skill Alignment Map</strong>
                        <p>Compare your skills against career targets to identify upskilling needs.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
                
        st.write("")
        
        col_f3, col_f4 = st.columns(2, gap="medium")
        with col_f3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-item">
                    <div class="feature-icon">📝</div>
                    <div class="feature-text">
                        <strong>Resume Profile Grader</strong>
                        <p>Evaluate your resume's competitive strength and industry readiness.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
                
        with col_f4:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-item">
                    <div class="feature-icon">🤖</div>
                    <div class="feature-text">
                        <strong>AI Interview Simulator</strong>
                        <p>Refine your communication and core knowledge with interactive live mock simulations.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.write("")
        
        # Corporate metrics footer banner
        st.markdown("""
        <div class="stats-banner" style="justify-content: center; text-align: center; max-width: 800px; margin: 40px auto 0 auto;">
            <div class="stat-box">
                <div class="stat-value">94.2%</div>
                <div class="stat-label">Placement Success</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">10k+</div>
                <div class="stat-label">Mock Interviews</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">50+</div>
                <div class="stat-label">Partner Schools</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.write("")
        
    else:
        # Centered Login Card
        st.write("")
        st.write("")
        _, col_login, _ = st.columns([1, 1.3, 1])
        
        with col_login:
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center; color: #FFFFFF; font-family: \"Outfit\", sans-serif; margin-bottom: 5px; margin-top: 0;'>Welcome Back</h3>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 13px; margin-bottom: 25px;'>Secure Access Portal</p>", unsafe_allow_html=True)
                
                role = st.selectbox("Select Account Role", ["Student", "Admin"])
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                st.write("")
                login_btn = st.button("🔑 Access Account Portal", use_container_width=True, type="primary")
                
                if login_btn:
                    if role == "Admin":
                        if username == "admin" and password == "admin123":
                            st.session_state.logged_in = True
                            st.session_state.user_role = "admin"
                            st.success("Access Granted! Loading Admin Workspace...")
                            st.rerun()
                        else:
                            st.error("Invalid Administrator Credentials.")
                    else:
                        if username == "student" and password == "student123":
                            st.session_state.logged_in = True
                            st.session_state.user_role = "student"
                            st.success("Access Granted! Loading Student Workspace...")
                            st.rerun()
                        else:
                            st.error("Invalid Student Credentials.")
                            
                st.write("---")
                st.markdown("""
                <div style='text-align: center; color: #94A3B8; font-size: 12.5px;'>
                    <b>Development Sandbox Demo Accounts</b><br/>
                    Admin: <code>admin</code> / <code>admin123</code><br/>
                    Student: <code>student</code> / <code>student123</code>
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
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-color: #0E1117;
        --card-bg: #1A1F2C;
        --card-border: #2D3748;
        --text-primary: #FFFFFF;
        --text-secondary: #A0AEC0;
        --accent: #6366F1;
        --accent-glow: rgba(99, 102, 241, 0.15);
    }
    
    html, body, [data-testid="stAppViewContainer"], .main * {
        font-family: 'Inter', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6, [class*="title"], [class*="header"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
    }
    
    /* Custom Scrollbar for Outstanding Visual Polish */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0E1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #2D3748;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #6366F1;
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
    
    /* Section containers - Styled Card with Colored Gradient Background */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, rgba(26, 31, 44, 0.85) 0%, rgba(15, 18, 30, 0.98) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        margin-bottom: 20px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 12px 35px rgba(99, 102, 241, 0.15) !important;
    }
    
    /* Ensure maximum visibility of text on colored dark backgrounds */
    div[data-testid="stVerticalBlockBorderWrapper"] h1, 
    div[data-testid="stVerticalBlockBorderWrapper"] h2, 
    div[data-testid="stVerticalBlockBorderWrapper"] h3, 
    div[data-testid="stVerticalBlockBorderWrapper"] h4, 
    div[data-testid="stVerticalBlockBorderWrapper"] h5, 
    div[data-testid="stVerticalBlockBorderWrapper"] h6 {
        color: #FFFFFF !important;
        margin-top: 5px !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"] p, 
    div[data-testid="stVerticalBlockBorderWrapper"] span,
    div[data-testid="stVerticalBlockBorderWrapper"] li,
    div[data-testid="stVerticalBlockBorderWrapper"] label,
    div[data-testid="stVerticalBlockBorderWrapper"] div {
        color: #E2E8F0 !important;
    }

    /* Style Streamlit Forms to match premium UI */
    div[data-testid="stForm"] {
        background: linear-gradient(135deg, rgba(30, 37, 54, 0.9) 0%, rgba(18, 22, 34, 0.98) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Customize Streamlit expanders */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        background-color: rgba(26, 31, 44, 0.6) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    /* Custom style for main sidebar menu list */
    [data-testid="stSidebar"] {
        background-color: #0E1117 !important;
        border-right: 1px solid #1A1F2C !important;
    }
</style>
"""

light_css = """
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-color: #F7FAFC;
        --card-bg: #FFFFFF;
        --card-border: #E2E8F0;
        --text-primary: #1A202C;
        --text-secondary: #4A5568;
        --accent: #4F46E5;
        --accent-glow: rgba(79, 70, 229, 0.1);
    }
    
    html, body, [data-testid="stAppViewContainer"], .main * {
        font-family: 'Inter', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6, [class*="title"], [class*="header"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
    }
    
    /* Custom Scrollbar for Outstanding Visual Polish */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #F7FAFC;
    }
    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #4F46E5;
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
    
    /* Section containers - Styled Card with Soft Pastel Indigo Background */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, rgba(238, 242, 255, 0.95) 0%, rgba(245, 247, 255, 0.98) 100%) !important;
        border: 1px solid rgba(79, 70, 229, 0.2) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.06) !important;
        margin-bottom: 20px !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(79, 70, 229, 0.4) !important;
        box-shadow: 0 12px 30px rgba(79, 70, 229, 0.12) !important;
    }
    
    /* Ensure maximum visibility of text on colored light backgrounds */
    div[data-testid="stVerticalBlockBorderWrapper"] h1, 
    div[data-testid="stVerticalBlockBorderWrapper"] h2, 
    div[data-testid="stVerticalBlockBorderWrapper"] h3, 
    div[data-testid="stVerticalBlockBorderWrapper"] h4, 
    div[data-testid="stVerticalBlockBorderWrapper"] h5, 
    div[data-testid="stVerticalBlockBorderWrapper"] h6 {
        color: #0F172A !important;
        margin-top: 5px !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"] p, 
    div[data-testid="stVerticalBlockBorderWrapper"] span,
    div[data-testid="stVerticalBlockBorderWrapper"] li,
    div[data-testid="stVerticalBlockBorderWrapper"] label,
    div[data-testid="stVerticalBlockBorderWrapper"] div {
        color: #1E293B !important;
    }

    /* Style Streamlit Forms to match premium UI */
    div[data-testid="stForm"] {
        background: linear-gradient(135deg, rgba(245, 247, 250, 0.95) 0%, rgba(250, 252, 255, 0.98) 100%) !important;
        border: 1px solid rgba(79, 70, 229, 0.25) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.08) !important;
    }
    
    /* Customize Streamlit expanders */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(79, 70, 229, 0.15) !important;
        background-color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    /* Custom style for main sidebar menu list */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
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
            "🤖 AI Interview Coach",
            "📊 Model Benchmarking",
            "💾 History & Data Management"
        ]
    else:
        nav_options = [
            "🔮 My Placement Prediction & XAI",
            "🎯 My Skill Gap & Roadmap",
            "📝 My Resume ATS Scorer",
            "🤖 My AI Interview Coach"
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
    elif "Interview" in nav_selection:
        routing_selection = "🤖 AI Interview Coach"
        
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
            with st.container(border=True):
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
            with st.container(border=True):
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
            with st.container(border=True):
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
            with st.container(border=True):
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
                st.session_state.show_prediction_results = True
                
                # Show results header
                st.success("🎉 Analysis complete! Scroll down to view diagnostic reports.")
                st.rerun()

        if st.session_state.get('show_prediction_results', False):
            student_dict = st.session_state.active_student_profile
            results = st.session_state.active_student_results
            name = student_dict['name']
            
            # 1. Prediction Overview Card
            with st.container(border=True):
                st.markdown("### 📊 Profile Assessment & Predicted Outcomes")
                
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
            
            # 2. Placement Probability Gauge
            with st.container(border=True):
                st.markdown("### 🎯 Placement Probability Gauge")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = float(results['placement_probability']),
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4F46E5"},
                        'bar': {'color': "#6366F1", 'thickness': 0.25},
                        'bgcolor': "rgba(255,255,255,0.05)" if st.session_state.theme_mode == 'dark' else "rgba(79,70,229,0.05)",
                        'borderwidth': 1,
                        'bordercolor': "#2D3748" if st.session_state.theme_mode == 'dark' else "#E2E8F0",
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                            {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.15)'},
                            {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                        ]
                    }
                ))
                fig_gauge.update_layout(
                    margin=dict(l=20, r=20, t=10, b=10),
                    height=240,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#FFFFFF" if st.session_state.theme_mode == 'dark' else "#1E293B"}
                )
                
                _, gc_col, _ = st.columns([1, 2, 1])
                with gc_col:
                    st.plotly_chart(fig_gauge, use_container_width=True)

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
            
            # 3. Explainable AI (SHAP Plots)
            with st.container(border=True):
                st.markdown("### 🧮 Explainable AI (XAI) Attribution")
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
            
            # 4. Career & Company Matcher
            with st.container(border=True):
                st.markdown("### 💼 Intelligent Career & Company Matcher")
                
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
        
        # Industry baselines for radar chart
        ROLE_BASELINES = {
            'Software Engineer': {'CGPA': 75, 'Aptitude': 70, 'Coding': 80, 'Communication': 70, 'Interview': 75},
            'Data Scientist': {'CGPA': 80, 'Aptitude': 75, 'Coding': 75, 'Communication': 75, 'Interview': 75},
            'Machine Learning Engineer': {'CGPA': 80, 'Aptitude': 80, 'Coding': 85, 'Communication': 70, 'Interview': 80},
            'Data Analyst': {'CGPA': 70, 'Aptitude': 75, 'Coding': 65, 'Communication': 80, 'Interview': 70},
            'Business Analyst': {'CGPA': 70, 'Aptitude': 80, 'Coding': 50, 'Communication': 90, 'Interview': 65},
            'AI Engineer': {'CGPA': 78, 'Aptitude': 80, 'Coding': 80, 'Communication': 75, 'Interview': 80},
            'Full Stack Developer': {'CGPA': 70, 'Aptitude': 70, 'Coding': 80, 'Communication': 75, 'Interview': 75}
        }
        
        # Analyze Gap
        gap_results = analyze_skill_gap(profile, target_role)
        
        # 1. Competency Gap Assessment Card
        with st.container(border=True):
            st.markdown(f"### 📊 Profile vs. Industry Benchmarks")
            
            col_gap1, col_gap2 = st.columns([1.1, 1.0])
            with col_gap1:
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
                # Radar chart plotting
                categories = ['Academic (CGPA)', 'Aptitude Score', 'Coding Competency', 'Communication', 'Tech Interview']
                student_values = [
                    float(profile.get('cgpa', 7.5)) * 10,
                    float(profile.get('aptitude', 70)),
                    float(profile.get('coding', 70)),
                    float(profile.get('communication', 70)),
                    float(profile.get('technical_interview', 70))
                ]
                
                baseline = ROLE_BASELINES.get(target_role, ROLE_BASELINES['Software Engineer'])
                baseline_values = [
                    baseline['CGPA'],
                    baseline['Aptitude'],
                    baseline['Coding'],
                    baseline['Communication'],
                    baseline['Interview']
                ]
                
                # Append first item to close radar loop
                categories.append(categories[0])
                student_values.append(student_values[0])
                baseline_values.append(baseline_values[0])
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=student_values,
                    theta=categories,
                    fill='toself',
                    name='Student Profile',
                    fillcolor='rgba(99, 102, 241, 0.25)',
                    line=dict(color='#6366F1', width=2)
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=baseline_values,
                    theta=categories,
                    fill='toself',
                    name=f'Industry Baseline',
                    fillcolor='rgba(245, 158, 11, 0.1)',
                    line=dict(color='#F59E0B', width=2, dash='dash')
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            color="#888888"
                        ),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    showlegend=True,
                    margin=dict(l=40, r=40, t=10, b=10),
                    height=280,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    font={'color': "#FFFFFF" if st.session_state.theme_mode == 'dark' else "#1E293B"}
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        # 2. Probability Boost Card
        with st.container(border=True):
            st.markdown("### 🚀 Placement Boost Forecast & Recommended Credentials")
            
            boost_col1, boost_col2 = st.columns([1, 1.2])
            with boost_col1:
                st.markdown(f"By closing these skill gaps, the student can increase their placement probability by:")
                st.markdown(f"<h1 style='color:#059669; font-size:48px; margin: 0;'>+{gap_results['estimated_probability_gain']}%</h1>", unsafe_allow_html=True)
            with boost_col2:
                st.markdown("##### 📜 Recommended Skill Certifications")
                for cert in gap_results['recommended_certifications']:
                    st.markdown(f"- **{cert['name']}** (Improves probability by **+{cert['prob_gain']}%**)")
                    
        st.write("---")
        
        # Personalized 12-Week Roadmap
        with st.container(border=True):
            st.markdown("### 📅 Tailored 12-Week Placement Action Plan")
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
        with st.container(border=True):
            st.subheader("Upload Resume PDF")
            uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
            
            # Select target role for resume scoring
            target_role = st.selectbox(
                "Target Career Track",
                options=["Software Engineer", "Data Scientist", "Machine Learning Engineer", "Data Analyst", "Business Analyst", "AI Engineer", "Full Stack Developer"]
            )
            
            analyze_button = st.button("📝 Analyze Resume ATS Score", use_container_width=True, disabled=(uploaded_file is None))
            
            if uploaded_file is not None and analyze_button:
                with st.spinner("Extracting PDF text and matching keywords..."):
                    # Extract text
                    resume_text = extract_text_from_pdf(uploaded_file)
                    # Analyze
                    resume_analysis = analyze_resume(resume_text, target_role)
                    # Store in session state
                    st.session_state.active_resume_analysis = resume_analysis
                    st.session_state.active_resume_filename = uploaded_file.name
                    st.session_state.active_resume_target = target_role
                    st.rerun()

    with col_res2:
        with st.container(border=True):
            st.subheader("Analysis Insights")
            
            # Check if we have cached results for the current uploaded file
            has_cache = (
                'active_resume_analysis' in st.session_state 
                and uploaded_file is not None 
                and st.session_state.active_resume_filename == uploaded_file.name
            )
            
            if not has_cache:
                st.info("Upload a student's resume PDF on the left and click 'Analyze' to view ATS keyword matches and scoring.")
            else:
                resume_analysis = st.session_state.active_resume_analysis
                score = resume_analysis['ats_score']
                score_color = "#EF4444" if score < 50 else "#F59E0B" if score < 75 else "#10B981"
                
                st.markdown(f"#### Target Career Track: `{st.session_state.active_resume_target}`")
                
                # ATS Score Gauge Chart
                fig_ats = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': score_color, 'thickness': 0.25},
                        'bgcolor': "rgba(255,255,255,0.05)" if st.session_state.theme_mode == 'dark' else "rgba(79,70,229,0.05)",
                        'borderwidth': 1
                    }
                ))
                fig_ats.update_layout(
                    margin=dict(l=20, r=20, t=10, b=10),
                    height=180,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#FFFFFF" if st.session_state.theme_mode == 'dark' else "#1E293B"}
                )
                st.plotly_chart(fig_ats, use_container_width=True)
                
                # Detailed stats
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
        with st.container(border=True):
            st.subheader("Classifier Suite Metrics Comparison Table")
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        st.write("---")
        
        # 2. ROC Curves & Confusion Matrices
        col_curve1, col_curve2 = st.columns(2)
        
        with col_curve1:
            with st.container(border=True):
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
            with st.container(border=True):
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
# MODULE 11: AI Interview Coach
# ==========================================
elif routing_selection == "🤖 AI Interview Coach":
    st.title("🤖 AI Interview Coach & Feedback Engine")
    st.markdown("Practice mock interview questions tailored to your career track and receive instant conceptual evaluations, matching analysis, and coaching guidelines.")
    
    from src.interview_coach import generate_ai_question, evaluate_ai_answer
    
    # Determine default track
    default_track = "Software Engineer"
    if 'active_student_profile' in st.session_state:
        from src.career_recommendation import recommend_careers
        recs = recommend_careers(st.session_state.active_student_profile)
        if recs:
            default_track = recs[0]['role']
            
    track_options = [
        "Software Engineer", "Data Scientist", "Machine Learning Engineer",
        "Data Analyst", "Business Analyst", "AI Engineer", "Full Stack Developer"
    ]
    
    selected_track = st.selectbox("Select Your Career Track", options=track_options, index=track_options.index(default_track) if default_track in track_options else 0)
    
    # Initialize session state for AI Interview
    if 'ai_interview_q_meta' not in st.session_state:
        st.session_state.ai_interview_q_meta = None
    if 'ai_previous_questions' not in st.session_state:
        st.session_state.ai_previous_questions = []
    if 'last_selected_ai_track' not in st.session_state:
        st.session_state.last_selected_ai_track = selected_track
        
    # Reset if track is changed
    if st.session_state.last_selected_ai_track != selected_track:
        st.session_state.ai_interview_q_meta = None
        st.session_state.ai_previous_questions = []
        st.session_state.last_selected_ai_track = selected_track
        if 'active_ai_review' in st.session_state:
            del st.session_state.active_ai_review
    
    # Next Question/Start Interview button
    btn_label = "⏭️ Start Live Interview" if st.session_state.ai_interview_q_meta is None else "⏭️ Load Next Unique Question"
    if st.button(btn_label, use_container_width=True):
        with st.spinner("Generating a unique question from Gemini AI..."):
            q_meta = generate_ai_question(selected_track, st.session_state.ai_previous_questions)
            if 'error' in q_meta:
                st.error(q_meta['error'])
            else:
                st.session_state.ai_interview_q_meta = q_meta
                st.session_state.ai_previous_questions.append(q_meta['question'])
                if 'active_ai_review' in st.session_state:
                    del st.session_state.active_ai_review
                st.rerun()
                
    if st.session_state.ai_interview_q_meta is None:
        st.info("💡 Click the button above to dynamically load an interview question from Gemini AI.")
    else:
        q_meta = st.session_state.ai_interview_q_meta
        
        st.info(f"🎙️ **Interview Question:** \n\n {q_meta['question']}")
        
        st.write("")
        
        user_ans = st.text_area(
            "Type or paste your response below (minimum 15-20 words recommended for structural feedback):",
            height=180,
            placeholder="Write your explanation here...",
            key=f"ai_ans_{selected_track}_{len(st.session_state.ai_previous_questions)}"
        )
        
        submit_btn = st.button("📝 Submit Answer for AI Review", use_container_width=True, type="primary")
        
        if submit_btn:
            if not user_ans.strip():
                st.warning("Please type a response before submitting.")
            else:
                with st.spinner("Gemini is evaluating your answer semantically..."):
                    review = evaluate_ai_answer(
                        selected_track,
                        q_meta['question'],
                        q_meta['concepts'],
                        user_ans
                    )
                    if 'error' in review:
                        st.error(review['error'])
                    else:
                        st.session_state.active_ai_review = review
                        st.rerun()
                        
        has_review = (
            'active_ai_review' in st.session_state
            and st.session_state.get('active_ai_review') is not None
            and 'error' not in st.session_state.active_ai_review
        )
        
        if has_review:
            review = st.session_state.active_ai_review
            score = review['score']
            
            st.write("---")
            st.markdown("### 📋 AI Assessment & Coaching Feedback")
            
            col_res1, col_res2 = st.columns([1, 1.2])
            
            with col_res1:
                with st.container(border=True):
                    st.markdown("##### 🎯 Semantic Coverage Score")
                    
                    fig_score = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': review['rating_color'], 'thickness': 0.25},
                            'bgcolor': "rgba(255,255,255,0.05)" if st.session_state.theme_mode == 'dark' else "rgba(79,70,229,0.05)",
                            'borderwidth': 1
                        }
                    ))
                    fig_score.update_layout(
                        margin=dict(l=20, r=20, t=10, b=10),
                        height=180,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#FFFFFF" if st.session_state.theme_mode == 'dark' else "#1E293B"}
                    )
                    st.plotly_chart(fig_score, use_container_width=True)
                    
                    st.markdown(f"**AI Readiness Rating:** <span style='color:{review['rating_color']}; font-weight:bold;'>{review['rating']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Answer Word Count:** {review['word_count']} words")
                    
            with col_res2:
                with st.container(border=True):
                    st.markdown("##### 🛠️ Concepts Match Assessment")
                    
                    st.write("**Identified Strengths / Covered Concepts:**")
                    if review['matched_concepts']:
                        tags_html = " ".join([f"<span style='background-color:rgba(16, 185, 129, 0.15); color:#10B981; border:1px solid rgba(16,185,129,0.3); padding:4px 10px; border-radius:12px; margin-right:6px; font-size:12px; font-weight:500; display:inline-block;'>{tag}</span>" for tag in review['matched_concepts']])
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.write("*No key concepts matched.*")
                        
                    st.write("")
                    st.write("**Missing Technical Elements:**")
                    if review['missing_concepts']:
                        tags_missing_html = " ".join([f"<span style='background-color:rgba(239, 68, 68, 0.15); color:#EF4444; border:1px solid rgba(239,68,68,0.3); padding:4px 10px; border-radius:12px; margin-right:6px; font-size:12px; font-weight:500; display:inline-block;'>{tag}</span>" for tag in review['missing_concepts']])
                        st.markdown(tags_missing_html, unsafe_allow_html=True)
                    else:
                        st.write("🎉 *None! You covered all expected concepts.*")
                        
            with st.container(border=True):
                st.markdown("##### 📋 Actionable Coaching Guidelines")
                for tip in review['guidelines']:
                    st.write(tip)
                    
            with st.expander("📚 Study Model Response & Explanation"):
                st.write("**Ideal Answer Structure:**")
                st.info(q_meta['ideal_response'])
                st.write("**Preparation Advice:**")
                st.write(q_meta['explanation'])

# ==========================================
# MODULE 14: History & Data Management
# ==========================================
elif routing_selection == "💾 History & Data Management":
    st.title("💾 History & Data Management")
    st.markdown("Upload new custom CSV datasets for preprocessing/validation, query past predictions from SQLite, or export history records.")
    
    tab1, tab2 = st.tabs(["📂 Upload & Validate Dataset", "📋 SQLite Prediction History"])
    
    with tab1:
        with st.container(border=True):
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
        with st.container(border=True):
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
