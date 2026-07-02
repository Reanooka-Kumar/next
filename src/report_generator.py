import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.colors import HexColor

def generate_student_pdf(
    filename, student_profile, prediction, career_recs, company_recs, skill_gap, roadmap
):
    """
    Generates a beautifully structured PDF analysis report for a student.
    """
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    story = []
    
    # -------------------------------------------------------------
    # 1. Colors & Typography Setup
    # -------------------------------------------------------------
    primary_color = HexColor("#1A237E")    # Deep Navy Blue
    secondary_color = HexColor("#006064")  # Deep Teal
    placed_green = HexColor("#E8F5E9")     # Light green background
    placed_text = HexColor("#2E7D32")      # Dark green text
    unplaced_red = HexColor("#FFEBEE")     # Light red background
    unplaced_text = HexColor("#C62828")    # Dark red text
    accent_gray = HexColor("#F5F5F5")      # Light grey
    text_color = HexColor("#37474F")        # Dark slate grey
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=5
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=12,
        textColor=HexColor("#546E7A"),
        spaceAfter=15
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=secondary_color,
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_text_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=text_color
    )
    
    bold_body_style = ParagraphStyle(
        'ReportBodyBold',
        parent=body_text_style,
        fontName='Helvetica-Bold'
    )
    
    cell_text_style = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=text_color
    )
    
    cell_header_style = ParagraphStyle(
        'CellHeader',
        parent=cell_text_style,
        fontName='Helvetica-Bold',
        textColor=colors.whitesmoke
    )
    
    # -------------------------------------------------------------
    # Header Banner Area
    # -------------------------------------------------------------
    header_data = [
        [
            Paragraph("<b>PLACEMENT INTELLIGENCE PLATFORM</b>", ParagraphStyle('H1', parent=title_style, textColor=colors.whitesmoke, fontSize=18)),
            Paragraph("Official Student Assessment Report<br/>Generated: 2026", ParagraphStyle('Sub', parent=subtitle_style, textColor=colors.lightgrey, alignment=2))
        ]
    ]
    header_table = Table(header_data, colWidths=[4.2*inch, 3.1*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), primary_color),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 15),
        ('RIGHTPADDING', (0,0), (-1,-1), 15),
    ]))
    
    story.append(header_table)
    story.append(Spacer(1, 15))
    
    # -------------------------------------------------------------
    # 2. Student Info & Placement Metric Cards
    # -------------------------------------------------------------
    # Student Details Box
    student_details_html = f"""
    <b>Name:</b> {student_profile.get('name', 'N/A')}<br/>
    <b>Age / Department:</b> {student_profile.get('age', 'N/A')} yrs | {student_profile.get('department', 'N/A')}<br/>
    <b>Academic Record:</b> CGPA {student_profile.get('cgpa', 'N/A')} | Attendance {student_profile.get('attendance', 'N/A')}%<br/>
    <b>Programming Profile:</b> {student_profile.get('languages', 'N/A')}
    """
    
    is_placed = prediction['placement_status'] == 'Placed'
    status_bg = placed_green if is_placed else unplaced_red
    status_txt = placed_text if is_placed else unplaced_text
    
    status_details_html = f"""
    <font size="10">PLACEMENT FORECAST</font><br/>
    <font size="16" color="{status_txt}"><b>{prediction['placement_status'].upper()}</b></font><br/>
    <b>Probability:</b> {prediction['placement_probability']:.1f}%<br/>
    <b>Confidence:</b> {prediction['confidence_score']:.1f}%
    """
    
    salary_details_html = f"""
    <font size="10">SALARY FORECAST (LPA)</font><br/>
    <font size="16" color="{secondary_color}"><b>{prediction['expected_salary']:.2f} LPA</b></font><br/>
    <b>Estimated Range:</b><br/>
    {prediction['salary_range_min']:.2f} - {prediction['salary_range_max']:.2f} LPA
    """
    
    cards_data = [
        [
            Paragraph(student_details_html, body_text_style),
            Paragraph(status_details_html, ParagraphStyle('StatusStyle', parent=body_text_style, alignment=1)),
            Paragraph(salary_details_html, ParagraphStyle('SalaryStyle', parent=body_text_style, alignment=1))
        ]
    ]
    
    cards_table = Table(cards_data, colWidths=[3.2*inch, 2.1*inch, 2.0*inch])
    cards_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), accent_gray),
        ('BACKGROUND', (1,0), (1,0), status_bg),
        ('BACKGROUND', (2,0), (2,0), accent_gray),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (0,0), 0.5, colors.lightgrey),
        ('BOX', (1,0), (1,0), 0.5, status_txt),
        ('BOX', (2,0), (2,0), 0.5, colors.lightgrey),
    ]))
    
    story.append(cards_table)
    story.append(Spacer(1, 15))
    
    # -------------------------------------------------------------
    # 3. Career Recommendations
    # -------------------------------------------------------------
    story.append(Paragraph("Target Career Recommendations", section_heading))
    
    # Take top 3 recommended roles
    career_table_data = [[
        Paragraph("Recommended Role", cell_header_style),
        Paragraph("Match Score", cell_header_style),
        Paragraph("Fit Level", cell_header_style),
        Paragraph("Key Reasons / Fit Criteria", cell_header_style)
    ]]
    
    for rec in career_recs[:3]:
        reasons_bullet = "<br/>".join([f"• {r}" for r in rec['reasons'][:2]])
        career_table_data.append([
            Paragraph(f"<b>{rec['role']}</b>", cell_text_style),
            Paragraph(f"{rec['score']:.1f}%", cell_text_style),
            Paragraph(rec['fit'], ParagraphStyle('FitT', parent=cell_text_style, textColor=placed_text if rec['score'] >= 75 else secondary_color)),
            Paragraph(reasons_bullet, cell_text_style)
        ])
        
    career_table = Table(career_table_data, colWidths=[1.8*inch, 1.0*inch, 1.1*inch, 3.4*inch])
    career_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), secondary_color),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, accent_gray]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(career_table)
    story.append(Spacer(1, 15))
    
    # -------------------------------------------------------------
    # 4. Company Recommendations
    # -------------------------------------------------------------
    story.append(Paragraph("Company Hiring Suitability", section_heading))
    
    # Take top 4 company matches
    company_table_data = [[
        Paragraph("Company Name", cell_header_style),
        Paragraph("Employer Category", cell_header_style),
        Paragraph("Suitability Fit", cell_header_style),
        Paragraph("Hiring Assessment", cell_header_style)
    ]]
    
    for comp in company_recs[:4]:
        company_table_data.append([
            Paragraph(f"<b>{comp['company']}</b>", cell_text_style),
            Paragraph(comp['tier'], cell_text_style),
            Paragraph(comp['fit'], ParagraphStyle('CompFit', parent=cell_text_style, textColor=placed_text if comp['fit'] == "Strong Match" else text_color)),
            Paragraph(comp['reasons'], cell_text_style)
        ])
        
    company_table = Table(company_table_data, colWidths=[1.5*inch, 1.8*inch, 1.2*inch, 2.8*inch])
    company_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, accent_gray]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(company_table)
    
    # Break to Page 2
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # 5. Skill Gap Analysis
    # -------------------------------------------------------------
    story.append(Paragraph("Competency & Skill Gap Analysis", section_heading))
    
    existing_skills_text = ", ".join(skill_gap['existing_skills']) if skill_gap['existing_skills'] else "None recorded"
    missing_skills_text = ", ".join(skill_gap['missing_skills'])
    
    gap_html = f"""
    <b>Target Career Path:</b> {skill_gap['target_role']}<br/>
    <b>Identified Priority Level:</b> <font color="{unplaced_text if skill_gap['priority_level'] == 'High' else secondary_color}"><b>{skill_gap['priority_level']}</b></font><br/>
    <b>Existing Strengths:</b> {existing_skills_text}<br/>
    <b>Identified Missing Skills:</b> <font color="{unplaced_text}"><b>{missing_skills_text}</b></font><br/>
    <b>Estimated Placement Gain on Gap Closure:</b> <font color="{placed_text}"><b>+{skill_gap['estimated_probability_gain']}%</b></font>
    """
    
    # Certifications Table
    cert_data = [[
        Paragraph("Recommended Skill Certification", cell_header_style),
        Paragraph("Estimated Chance Boost", cell_header_style)
    ]]
    for cert in skill_gap['recommended_certifications'][:3]:
        cert_data.append([
            Paragraph(cert['name'], cell_text_style),
            Paragraph(f"+{cert['prob_gain']}% chance", ParagraphStyle('PBoost', parent=cell_text_style, textColor=placed_text))
        ])
        
    cert_table = Table(cert_data, colWidths=[2.5*inch, 1.2*inch])
    cert_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), secondary_color),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    
    gap_summary_data = [
        [
            Paragraph(gap_html, body_text_style),
            cert_table
        ]
    ]
    
    gap_outer_table = Table(gap_summary_data, colWidths=[3.5*inch, 3.8*inch])
    gap_outer_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), accent_gray),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (0,0), 0.5, colors.lightgrey),
    ]))
    story.append(gap_outer_table)
    story.append(Spacer(1, 15))
    
    # -------------------------------------------------------------
    # 6. Personal Learning Roadmap (12 Weeks)
    # -------------------------------------------------------------
    story.append(Paragraph(f"Personalized 12-Week Preparation Schedule", section_heading))
    
    roadmap_table_data = [[
        Paragraph("Timeline", cell_header_style),
        Paragraph("Weekly Study Focus & Key Topics", cell_header_style),
        Paragraph("Suggested Learning Resources", cell_header_style)
    ]]
    
    for block in roadmap:
        actions_list = "<br/>".join([f"• {act}" for act in block['action_items']])
        roadmap_table_data.append([
            Paragraph(f"<b>{block['weeks']}</b><br/><font size='8' color='#546E7A'>{block['topic']}</font>", cell_text_style),
            Paragraph(actions_list, cell_text_style),
            Paragraph(block['resources'], ParagraphStyle('Res', parent=cell_text_style, fontName='Helvetica-Oblique'))
        ])
        
    roadmap_table = Table(roadmap_table_data, colWidths=[1.8*inch, 3.8*inch, 1.7*inch])
    roadmap_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, accent_gray]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(roadmap_table)
    
    # Footer disclaimer
    story.append(Spacer(1, 20))
    disclaimer_text = """
    <i>Disclaimer: This assessment is generated algorithmically by the Placement Intelligence Platform
    using ensemble machine learning models trained on historical collegiate recruitment datasets. 
    It is intended solely for training, academic diagnostics, and skill enhancement advisory purposes.</i>
    """
    story.append(Paragraph(disclaimer_text, ParagraphStyle('Disclaimer', parent=body_text_style, fontSize=7.5, leading=10, textColor=HexColor("#78909C"), alignment=1)))
    
    # Build Document
    doc.build(story)
