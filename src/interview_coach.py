import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

def get_gemini_client(api_key=None):
    """
    Initializes the Google Gemini API configuration.
    Looks for the key in:
    1. The passed argument (from UI/User input)
    2. Environmental variable GEMINI_API_KEY
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if key:
        genai.configure(api_key=key.strip())
        return True
    return False

def generate_ai_question(role, previous_questions=None, api_key=None):
    """
    Generates a unique, role-specific interview question using Gemini API.
    To maintain absolute uniqueness across sessions, it uses the previous_questions list
    to instruct the model to avoid generating duplicates.
    """
    if previous_questions is None:
        previous_questions = []
        
    if not get_gemini_client(api_key):
        return {
            'error': "AI API key is not configured. Please check your environment configurations."
        }
        
    previous_qs_filter = ""
    if previous_questions:
        previous_qs_filter = "Please avoid generating any of the following questions that were already asked in this session:\n" + "\n".join([f"- {q}" for q in previous_questions])
        
    prompt = f"""
You are an elite corporate technical interviewer and talent acquisition specialist.
Generate one unique, challenging, and role-appropriate interview question for the career track: "{role}".

{previous_qs_filter}

Respond strictly in JSON format matching the schema below:
{{
  "question": "The generated interview question text.",
  "concepts": [
    "Core technical concept/terminology expected in answer 1",
    "Core technical concept/terminology expected in answer 2",
    "Core technical concept/terminology expected in answer 3",
    "Core technical concept/terminology expected in answer 4"
  ],
  "ideal_response": "A brief model answer demonstrating the target depth and correct concepts.",
  "explanation": "Practical advice for the candidate on how to structure their answer and what the interviewer evaluates."
}}

Do not include any code block syntax markers (like ```json), markdown formatting, or trailing text. Return only the raw JSON string.
"""
    try:
        model = genai.GenerativeModel('gemini-3.5-flash')
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Strip any formatting fences if returned by the model
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                lines = lines[1:-1]
            text = "\n".join(lines).strip()
            
        data = json.loads(text)
        return data
    except Exception as e:
        return {
            'error': f"AI Question Generation Failed: {str(e)}"
        }

def evaluate_ai_answer(role, question, expected_concepts, user_answer, api_key=None):
    """
    Evaluates a candidate's response using the Gemini LLM for semantic checks,
    grading alignment, and detailed coaching guidelines.
    """
    if not get_gemini_client(api_key):
        return {
            'error': "AI API key is not configured. Please check your environment configurations."
        }
        
    concepts_str = ", ".join(expected_concepts)
    word_count = len(user_answer.split())
    
    prompt = f"""
You are a senior hiring manager conducting an evaluation of a candidate's answer.

Role Track: "{role}"
Interview Question: "{question}"
Expected Core Concepts: [{concepts_str}]
Candidate's Answer: "{user_answer}"

Evaluate the response semantically based on technical accuracy, clarity, and completeness.
Check if they addressed the expected concepts (or synonyms/related correct details). 
If the answer is extremely short (e.g. less than 15-20 words), penalize the score heavily since it lacks professional depth.

Respond strictly in JSON format matching the schema below:
{{
  "score": 85.5, // Numeric percentage score from 0.0 to 100.0
  "rating": "Readiness description, e.g. Exceptional (Strong Match), Needs Practice (Moderate Coverage), or Beginner (Insufficient Details)",
  "rating_color": "green for score >= 80, orange for score 50-79, red for score < 50",
  "matched_concepts": [
    "Expected technical concept matched in candidate's response",
    ...
  ],
  "missing_concepts": [
    "Expected technical concept candidate failed to touch upon",
    ...
  ],
  "guidelines": [
    "Direct feedback item 1",
    "Direct feedback item 2"
  ]
}}

Do not include any code block syntax markers (like ```json), markdown formatting, or trailing text. Return only the raw JSON string.
"""
    try:
        model = genai.GenerativeModel('gemini-3.5-flash')
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Strip code block boundaries if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                lines = lines[1:-1]
            text = "\n".join(lines).strip()
            
        data = json.loads(text)
        data['word_count'] = word_count
        return data
    except Exception as e:
        return {
            'error': f"AI Response Evaluation Failed: {str(e)}"
        }
