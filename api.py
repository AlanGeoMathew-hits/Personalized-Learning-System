from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

# Optional Gemini integration
USE_GEMINI = True
try:
    import google.generativeai as genai
except Exception:
    USE_GEMINI = False

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# ---- Load mo  del + columns ----
MODEL_PATH = "risk_predictor_model.joblib"
COLS_PATH = "model_columns.joblib"

if not (os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH)):
    raise FileNotFoundError(
        "Model files not found. Run a training script first to create "
        "'risk_predictor_model.joblib' and 'model_columns.joblib'."
    )

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLS_PATH)


def get_gemini_recommendations(student_name, student_class, is_at_risk, df_row):
    """Return Markdown string of recommendations."""
    if not USE_GEMINI:
        return (
            "### AI Advisor Offline\n"
            "Gemini is not available. Set USE_GEMINI=True and provide GEMINI_API_KEY to enable suggestions."
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "### AI Advisor Offline\n**Error:** GEMINI_API_KEY not found in environment variables."

    persona = (
        f"The student's name is {student_name}, in {student_class}, with {df_row['Motivation_Level'].iloc[0]} motivation "
        f"and {df_row['Parental_Involvement'].iloc[0]} parental involvement."
    )
    if is_at_risk:
        weaknesses = []
        if df_row['Previous_Scores'].iloc[0] < 75: weaknesses.append("low previous scores")
        if df_row['Attendance'].iloc[0] < 80: weaknesses.append("low attendance")
        prompt_subject = f"The student has been identified as 'at-risk' due to: {', '.join(weaknesses)}."
        prompt_goal = "Provide empathetic, holistic strategies to improve. For each, include a relevant YouTube link."
        ai_header = f"AI-Powered Action Plan for {student_name}"
    else:
        prompt_subject = "The student has been identified as 'not at-risk'."
        prompt_goal = "Provide creative enrichment strategies to excel further, including a YouTube Masterclass plan."
        ai_header = f"AI-Powered Enrichment Plan for {student_name}"

    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""Act as an expert academic coach named Aura for {student_name}.
        **Context:** {persona}
        **Analysis:** {prompt_subject}
        **Task:** {prompt_goal}
        Structure your response in Markdown with the header "### {ai_header}", sub-headers, and bullet points."""
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"### Error Connecting to AI Advisor\nAn error occurred: {e}"


@app.route("/predict", methods=["POST"])
def predict():
    """Takes student data, returns prediction and recommendations."""
    if not request.json:
        return jsonify({"error": "Invalid input: no JSON received"}), 400

    data = request.json

    # 1) Build a DataFrame from incoming data
    required_fields = ["hours_studied", "attendance", "sleep_hours", "previous_scores", "tutoring_sessions",
                       "physical_activity", "parental_involvement", "extracurricular_activities", "motivation_level",
                       "parental_education_level", "teacher_quality", "peer_influence", "gender"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Invalid input: missing one or more required fields"}), 400

    input_data = {
        'Hours_Studied': data['hours_studied'], 'Attendance': data['attendance'],
        'Sleep_Hours': data['sleep_hours'], 'Previous_Scores': data['previous_scores'],
        'Tutoring_Sessions': data['tutoring_sessions'], 'Physical_Activity': data['physical_activity'],
        'Parental_Involvement': data['parental_involvement'],
        'Extracurricular_Activities': data['extracurricular_activities'],
        'Motivation_Level': data['motivation_level'], 'Parental_Education_Level': data['parental_education_level'],
        'Teacher_Quality': data['teacher_quality'], 'Peer_Influence': data['peer_influence'],
        'Gender': data['gender'],
        # These can be defaulted as they were less critical in the model
        'Internet_Access': 'Yes', 'School_Type': 'Public', 'Family_Income': 'Medium',
        'Access_to_Resources': 'Medium', 'Learning_Disabilities': 'No', 'Distance_from_Home': 'Near'
    }
    df = pd.DataFrame([input_data])

    # 2) One-hot encode using get_dummies and align to trained columns
    df_proc = pd.get_dummies(df)
    df_aligned = df_proc.reindex(columns=model_columns, fill_value=0)

    # 3) Predict
    proba = model.predict_proba(df_aligned)[0]
    pred = int(proba.argmax())
    confidence = float(proba[pred])
    is_at_risk = (pred == 1)

    # 4) Recommendations (Markdown)
    recommendations = get_gemini_recommendations(
        data.get('name', 'the student'),
        data.get('grade', 'Class'),
        is_at_risk,
        df
    )

    return jsonify({
        "is_at_risk": is_at_risk,
        "confidence": confidence,
        "recommendations": recommendations,
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)