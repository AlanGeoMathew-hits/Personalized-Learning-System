from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# --- Load Model ---
try:
    model = joblib.load('risk_predictor_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
except FileNotFoundError:
    print("FATAL ERROR: Model files not found. Please run 'save_model.py' first.")
    exit()


# --- Gemini API Call with UPGRADED Prompt ---
def get_gemini_recommendations(student_name, student_class, is_at_risk, student_data):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "AI Advisor Offline. GEMINI_API_KEY not found in environment variables."

    persona = (
        f"The student's name is {student_name}, a high-achieving student in {student_class} "
        f"with an excellent academic record (previous score: {student_data['Previous_Scores'].iloc[0]}) "
        f"and a {student_data['Motivation_Level'].iloc[0]} motivation level."
    )

    if is_at_risk:
        weaknesses = []
        if student_data['Previous_Scores'].iloc[0] < 75: weaknesses.append("low previous scores")
        if student_data['Attendance'].iloc[0] < 80: weaknesses.append("low attendance")
        prompt_subject = f"The student has been identified as 'at-risk' due to: {', '.join(weaknesses)}."
        prompt_goal = "Provide empathetic, holistic, and actionable strategies to help them improve. For each strategy, you must find and include a relevant, high-quality YouTube video link formatted as a Markdown link, e.g., `[Video Title](URL)`."
        ai_header = f"AI-Powered Action Plan for {student_name}"
    else:
        # --- THIS IS THE NEW, UPGRADED PROMPT FOR THE ENRICHMENT PLAN ---
        prompt_subject = "The student has been identified as 'not at-risk' and is ready for an advanced challenge."
        prompt_goal = (
            "Your goal is to design a 'YouTube Masterclass' for them. "
            "First, create a compelling title for the masterclass. "
            "Then, find a sequence of 3 high-quality YouTube videos that create a learning pathway on an advanced topic relevant to their grade level. "
            "The pathway should logically progress from a core concept to a more advanced application. "
            "For each video, you must provide the Markdown link (`[Video Title](URL)`) and a one-sentence summary of what the student will learn from it."
        )
        ai_header = f"AI-Curated YouTube Masterclass for {student_name}"
        # --------------------------------------------------------------------

    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""
        Act as an expert, innovative academic coach named Aura for {student_name}.
        **Context:** {persona}
        **Analysis:** {prompt_subject}
        **Your Task:** {prompt_goal}
        Structure your response in Markdown with the main header "### {ai_header}".
        Use bolding and a numbered list for the videos.
        Keep the tone inspiring and focused on pushing boundaries.
        """
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to AI Advisor: {e}"


# --- API Endpoint (No changes needed) ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = {
        'Hours_Studied': data['hours_studied'], 'Attendance': data['attendance'],
        'Sleep_Hours': data['sleep_hours'], 'Previous_Scores': data['previous_scores'],
        'Tutoring_Sessions': data['tutoring_sessions'], 'Physical_Activity': data['physical_activity'],
        'Parental_Involvement': data['parental_involvement'],
        'Extracurricular_Activities': data['extracurricular_activities'],
        'Motivation_Level': data['motivation_level'], 'Parental_Education_Level': data['parental_education_level'],
        'Teacher_Quality': data['teacher_quality'], 'Peer_Influence': data['peer_influence'],
        'Internet_Access': 'Yes', 'School_Type': 'Public', 'Family_Income': 'Medium', 'Gender': data['gender'],
        'Access_to_Resources': 'Medium', 'Learning_Disabilities': 'No', 'Distance_from_Home': 'Near'
    }
    input_df = pd.DataFrame([input_data])
    input_df_processed = pd.get_dummies(input_df)
    input_df_aligned = input_df_processed.reindex(columns=model_columns, fill_value=0)

    prediction_proba = model.predict_proba(input_df_aligned)[0]
    prediction = int(prediction_proba.argmax())
    confidence = float(prediction_proba[prediction])

    is_at_risk = (prediction == 1)
    recommendations = get_gemini_recommendations(data['name'], data['grade'], is_at_risk, input_df)

    return jsonify({
        'is_at_risk': is_at_risk,
        'confidence': confidence,
        'recommendations': recommendations
    })


# --- Run Server (No changes needed) ---
if __name__ == '__main__':
    app.run(debug=True)