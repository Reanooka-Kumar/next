import os
import json
import random
import google.generativeai as genai
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

# Offline high-quality question bank for all 7 target career tracks
MOCK_QUESTIONS = {
    'Software Engineer': [
        {
            'question': "Explain the difference between a process and a thread, and how they share memory.",
            'concepts': ["process", "thread", "virtual memory", "address space", "stack"],
            'ideal_response': "A process is an independent execution unit with its own virtual memory space, whereas a thread is a lightweight execution unit inside a process that shares the process's memory space. Threads share the heap, global variables, but have their own stack.",
            'explanation': "Interviewer evaluates understanding of operating system resource allocation and concurrency fundamentals."
        },
        {
            'question': "Describe how a hash table works and explain how collision resolution is handled.",
            'concepts': ["hash function", "collision", "chaining", "open addressing", "probing"],
            'ideal_response': "A hash table uses a hash function to map keys to bucket indices for O(1) average lookup. Collisions are resolved using chaining (linked lists) or open addressing (probing).",
            'explanation': "Focus on hash function distribution, load factor, and worst-case time complexity."
        }
    ],
    'Data Scientist': [
        {
            'question': "What is the difference between L1 and L2 regularization, and when would you use each?",
            'concepts': ["L1 Lasso", "L2 Ridge", "sparsity", "multicollinearity", "overfitting"],
            'ideal_response': "L1 regularization (Lasso) penalizes the absolute sum of coefficients, creating sparse models for feature selection. L2 regularization (Ridge) penalizes the squared sum, shrinking coefficients to mitigate multicollinearity.",
            'explanation': "Demonstrate understanding of the bias-variance tradeoff and parameter shrinkage."
        },
        {
            'question': "Explain the difference between bagging and boosting, giving examples of algorithms for each.",
            'concepts': ["bagging", "boosting", "parallel", "sequential", "variance", "bias", "Random Forest", "XGBoost"],
            'ideal_response': "Bagging builds independent models in parallel to reduce variance (e.g. Random Forest). Boosting builds models sequentially where each model corrects previous errors to reduce bias (e.g. XGBoost, AdaBoost).",
            'explanation': "Explain how bootstrap aggregating works and contrast it with sequential gradient updates."
        }
    ],
    'Machine Learning Engineer': [
        {
            'question': "Describe the vanishing gradient problem in deep neural networks and how to mitigate it.",
            'concepts': ["vanishing gradient", "backpropagation", "activation function", "ReLU", "residual connections", "LSTM"],
            'ideal_response': "During backpropagation, gradients shrink exponentially as they propagate back to early layers, preventing weight updates. Mitigation includes using ReLU activation, batch normalization, residual connections, or LSTM units.",
            'explanation': "Explain the math behind chain rule multiplication of small derivatives."
        },
        {
            'question': "What is overfitting, and how do you diagnose and prevent it in ML models?",
            'concepts': ["overfitting", "validation split", "early stopping", "dropout", "regularization", "data augmentation"],
            'ideal_response': "Overfitting is when a model learns training noise instead of the general pattern. Diagnose it via diverging train/val loss curves. Prevent it using dropout, regularization, early stopping, and data augmentation.",
            'explanation': "Focus on model generalization capacity and complexity control."
        }
    ],
    'Data Analyst': [
        {
            'question': "What is the difference between INNER JOIN, LEFT JOIN, and RIGHT JOIN in SQL?",
            'concepts': ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "NULL", "intersection", "matching records"],
            'ideal_response': "INNER JOIN returns rows with matching keys in both tables. LEFT JOIN returns all rows from the left table and matched rows from the right, with NULLs for unmatched. RIGHT JOIN is the inverse.",
            'explanation': "Explain relational data operations and how NULL values behave in result sets."
        },
        {
            'question': "Describe how you would clean a dataset with missing values and outliers.",
            'concepts': ["imputation", "median", "z-score", "IQR", "outliers", "drop"],
            'ideal_response': "Clean missing values via mean/median imputation, business logic overrides, or drop rows. Clean outliers by identifying them with IQR/Z-score thresholds and replacing or trimming them.",
            'explanation': "Show understanding of data distribution preservation and handling skewed metrics."
        }
    ],
    'Business Analyst': [
        {
            'question': "What is the difference between functional and non-functional requirements?",
            'concepts': ["functional", "non-functional", "system behavior", "performance", "security", "scalability"],
            'ideal_response': "Functional requirements define what the system must do (e.g., user actions). Non-functional requirements define how the system must perform (e.g. scalability, security, page load speed).",
            'explanation': "Focus on operational constraints vs transactional actions."
        },
        {
            'question': "Explain the difference between Agile and Waterfall project methodologies.",
            'concepts': ["Agile", "Waterfall", "iterative", "sequential", "sprint", "scope change"],
            'ideal_response': "Waterfall is a linear, sequential phase approach (design, build, test). Agile is iterative, dividing work into short sprints with high adaptability to scope changes.",
            'explanation': "Focus on customer feedback cycles, delivery frequency, and risk management."
        }
    ],
    'AI Engineer': [
        {
            'question': "What is Retrieval-Augmented Generation (RAG) and why is it used with Large Language Models?",
            'concepts': ["RAG", "vector database", "embeddings", "hallucination", "external knowledge", "context window"],
            'ideal_response': "RAG retrieves relevant document chunks from an external database using vector embeddings, prepending them to the LLM prompt. This reduces hallucinations and incorporates dynamic, private information.",
            'explanation': "Focus on token efficiency, real-time knowledge updates, and embedding retrieval mechanics."
        },
        {
            'question': "Explain the difference between fine-tuning a model and prompt engineering.",
            'concepts': ["fine-tuning", "prompt engineering", "weights", "in-context learning", "dataset", "compute resources"],
            'ideal_response': "Prompt engineering designs inputs to guide an existing model's responses without weight changes. Fine-tuning trains model weights on a task-specific dataset, requiring compute resources.",
            'explanation': "Contrast token usage costs, task alignment, and model weight adjustment trade-offs."
        }
    ],
    'Full Stack Developer': [
        {
            'question': "Explain the difference between client-side rendering (CSR) and server-side rendering (SSR).",
            'concepts': ["client-side", "server-side", "SEO", "hydration", "initial load time", "bundle size"],
            'ideal_response': "CSR sends a minimal HTML shell and loads javascript to build the DOM in the browser. SSR renders the full HTML on the server first, improving SEO and initial page speed before hydration.",
            'explanation': "Discuss page indexing efficiency, server resources, and user interaction latency."
        },
        {
            'question': "What is REST and how does it compare to GraphQL API architectures?",
            'concepts': ["REST", "GraphQL", "over-fetching", "endpoint", "query language", "single request"],
            'ideal_response': "REST accesses resources via multiple HTTP endpoints (GET, POST). GraphQL uses a single endpoint with a query language, allowing clients to request exactly what they need to avoid over-fetching.",
            'explanation': "Highlight schema maintenance, network request latency, and caching differences."
        }
    ]
}

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
    Generates a unique, role-specific interview question using the LLM API.
    To maintain absolute uniqueness across sessions, it uses the previous_questions list
    to instruct the model to avoid generating duplicates.
    """
    if previous_questions is None:
        previous_questions = []
        
    # Check key and try to call the API
    has_key = get_gemini_client(api_key)
    
    if not has_key:
        # Fallback to local question database if no key
        return get_local_fallback_question(role, previous_questions)
        
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
        # Graceful fallback on API rate limit (429) or other request exceptions
        return get_local_fallback_question(role, previous_questions)

def get_local_fallback_question(role, previous_questions):
    role_questions = MOCK_QUESTIONS.get(role, MOCK_QUESTIONS['Software Engineer'])
    available = [q for q in role_questions if q['question'] not in previous_questions]
    if not available:
        available = role_questions
    chosen = random.choice(available)
    return {
        'question': chosen['question'],
        'concepts': chosen['concepts'],
        'ideal_response': chosen['ideal_response'],
        'explanation': chosen['explanation'] + " (Offline Simulator Mode due to rate limits/API configuration)"
    }

def evaluate_ai_answer(role, question, expected_concepts, user_answer, api_key=None):
    """
    Evaluates a candidate's response using LLM API for semantic checks.
    """
    has_key = get_gemini_client(api_key)
    if not has_key:
        return evaluate_ai_answer_offline(role, question, expected_concepts, user_answer)
        
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
        # Graceful fallback on API rate limit (429) or other request exceptions
        return evaluate_ai_answer_offline(role, question, expected_concepts, user_answer)

def evaluate_ai_answer_offline(role, question, expected_concepts, user_answer):
    cleaned_answer = user_answer.lower().replace(".", " ").replace(",", " ").replace(";", " ")
    answer_words = cleaned_answer.split()
    
    matched = []
    missing = []
    for concept in expected_concepts:
        concept_lower = concept.lower()
        if concept_lower in cleaned_answer:
            matched.append(concept)
        else:
            # Check individual key words of the concept to make it smarter
            parts = concept_lower.split()
            if any(p in answer_words for p in parts if len(p) > 3):
                matched.append(concept)
            else:
                missing.append(concept)
                
    word_count = len(user_answer.split())
    concept_ratio = len(matched) / len(expected_concepts) if expected_concepts else 1.0
    base_score = concept_ratio * 100.0
    
    if word_count < 10:
        base_score = max(5.0, base_score * 0.2)
    elif word_count < 20:
        base_score = max(15.0, base_score * 0.5)
    elif word_count > 150:
        base_score = min(100.0, base_score + 5.0)
        
    score = round(base_score, 1)
    
    if score >= 80:
        rating = "Exceptional (Strong Match)"
        rating_color = "green"
    elif score >= 50:
        rating = "Needs Practice (Moderate Coverage)"
        rating_color = "orange"
    else:
        rating = "Beginner (Insufficient Details)"
        rating_color = "red"
        
    guidelines = []
    if word_count < 20:
        guidelines.append("Your response is brief. Expand your answer to explain technical details and support your statements.")
    if missing:
        guidelines.append(f"Incorporate key industry concepts like: {', '.join(missing[:3])}.")
    if matched:
        guidelines.append(f"Nice job addressing {', '.join(matched[:2])} in your explanation.")
    else:
        guidelines.append("Try to define core terms and structure your answer with clear definitions.")
        
    guidelines.append("Structure your responses with standard structured frameworks or state raw definitions for conceptual ones.")
    guidelines.append("(Evaluation processed in offline fallback mode)")
    
    return {
        'score': score,
        'rating': rating,
        'rating_color': rating_color,
        'matched_concepts': matched,
        'missing_concepts': missing,
        'guidelines': guidelines,
        'word_count': word_count
    }
