import streamlit as st
import pandas as pd
import joblib
import time
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(
    page_title="Aura Predict: Student Success AI",
    page_icon="🔮",
    layout="wide"
)


# --- Custom CSS for Styling ---
def embed_css():
    css = """
    .stApp {
        background: url("https://images.unsplash.com/photo-1522202176988-66273c2fd55f?q=80&w=2940&auto=format&fit=crop") no-repeat center center fixed;
        background-size: cover;
    }
    .main .block-container { padding-top: 2rem; padding-left: 5rem; padding-right: 5rem; }
    .result-card { background-color: rgba(42, 50, 68, 0.85); border-radius: 20px; padding: 25px; margin-top: 20px; border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(15px); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); }
    .result-card h2 { font-size: 24px; font-weight: bold; margin-bottom: 5px; }
    .result-card .status-safe { color: #28a745; }
    .result-card .status-risk { color: #dc3545; }
    .result-card p, .result-card li { font-size: 16px; line-height: 1.6; }
    .result-card .ai-section { border-top: 1px solid rgba(255, 255, 255, 0.2); margin-top: 20px; padding-top: 15px; }
    .result-card .ai-section h3 { color: #4F8BF9; font-size: 20px; margin-bottom: 10px; }
    .stButton>button { border-radius: 20px; border: 1px solid #4F8BF9; background-color: #4F8BF9; color: white; width: 100%; height: 3em; margin-top: 20px;}
    h1, h2, h3, h4, h5, h6, p, .stTextInput>label, .stSelectbox>label, .stExpander>summary { color: #FFFFFF; }
    .stExpander { background-color: rgba(14, 17, 23, 0.8); border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); }
    """
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


embed_css()


# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('risk_predictor_model.joblib')
        columns = joblib.load('model_columns.joblib')
        return model, columns
    except FileNotFoundError:
        return None, None


model, model_columns = load_model()

if model is None:
    st.error("Model files not found. Please run 'save_model.py' first.")
    st.stop()

# --- API Key Handling with Fallback ---
api_key = None
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.sidebar.warning("`secrets.toml` not found. Please enter your API key below to enable AI features.", icon="⚠️")
    api_key = st.sidebar.text_input("Enter your Gemini API Key here:", type="password")


# --- Gemini API Call Function (Updated for Both Cases) ---
def get_gemini_recommendations(student_name, student_class, is_at_risk, student_data):
    """Generates tailored recommendations from Gemini for both at-risk and not-at-risk students."""
    if not api_key:
        return '<div class="ai-section"><h3>AI Advisor Offline</h3><p>Please enter a valid Gemini API key to receive AI-powered recommendations.</p></div>'

    # Prepare prompt based on risk status
    if is_at_risk:
        weaknesses = []
        if student_data['Previous_Scores'].iloc[0] < 75: weaknesses.append("low previous scores")
        if student_data['Attendance'].iloc[0] < 80: weaknesses.append("low attendance")
        if student_data['Hours_Studied'].iloc[0] < 20: weaknesses.append("insufficient study hours")
        if student_data['Motivation_Level'].iloc[0] == 'Low': weaknesses.append("low motivation")

        prompt_subject = f"The student's main challenges are: {', '.join(weaknesses)}."
        prompt_goal = "Provide encouraging, specific, and actionable strategies to help them improve in these areas."
        ai_header = f"### ✨ AI-Powered Strategies for {student_name}"

    else:  # If Not At-Risk
        strengths = []
        if student_data['Previous_Scores'].iloc[0] >= 90: strengths.append("excellent previous scores")
        if student_data['Attendance'].iloc[0] >= 95: strengths.append("high attendance")
        if student_data['Hours_Studied'].iloc[0] >= 25: strengths.append("dedicated study habits")

        prompt_subject = f"The student shows strong potential with strengths in: {', '.join(strengths)}."
        prompt_goal = "Provide creative and challenging enrichment strategies to help them excel further, explore advanced topics, and develop leadership skills."
        ai_header = f"### ✨ AI-Powered Enrichment Plan for {student_name}"

    try:
        genai.configure(api_key=api_key)

        # --- THIS IS THE CORRECTED LINE ---
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        # ---------------------------------

        prompt = f"""
        Act as an expert academic advisor named Aura for a student named {student_name} in {student_class}.
        {prompt_subject}
        {prompt_goal}

        Structure your response in Markdown with the main header "{ai_header}".
        Then, create relevant sub-headers (e.g., "#### On Deepening Knowledge:") and provide 2-3 specific, actionable bullet points.
        Keep the tone positive and empowering.
        """
        response = gemini_model.generate_content(prompt)
        return f'<div class="ai-section">{response.text}</div>'
    except Exception as e:
        return f'<div class="ai-section"><h3>Error Connecting to AI Advisor</h3><p>There was an issue with the API call. Please check your API key and try again.</p><p><i>Error details: {e}</i></p></div>'


# --- Feedback Generation Function (Updated to call Gemini for both cases) ---
def generate_feedback(student_data, prediction, student_name, student_class):
    is_at_risk = (prediction == 1)

    # Get AI recommendations regardless of risk status
    ai_html = get_gemini_recommendations(student_name, student_class, is_at_risk, student_data)

    if not is_at_risk:
        return (
            f'<div class="result-card">'
            f'<h2 class="status-safe">✅ Analysis for {student_name} (Class {student_class})</h2>'
            f'<p><b>Prediction:</b> Not At-Risk</p>'
            f'<p><b>Summary:</b> Excellent work, {student_name}! Your profile shows strong indicators for success. You have a great foundation to build upon.</p>'
            f'{ai_html}'  # Add AI enrichment plan
            '</div>'
        )
    else:
        feedback_points = []
        if student_data['Previous_Scores'].iloc[0] < 75: feedback_points.append(
            "<li><b>Academics:</b> Previous scores suggest a need to reinforce foundational concepts.</li>")
        if student_data['Attendance'].iloc[0] < 80: feedback_points.append(
            "<li><b>Engagement:</b> Attendance is a critical area for improvement.</li>")
        if student_data['Hours_Studied'].iloc[0] < 20: feedback_points.append(
            "<li><b>Study Habits:</b> Current study time may be insufficient.</li>")
        if student_data['Motivation_Level'].iloc[0] == 'Low': feedback_points.append(
            "<li><b>Well-being:</b> Low motivation can be a barrier.</li>")

        return (
            f'<div class="result-card">'
            f'<h2 class="status-risk">⚠️ Analysis for {student_name} (Class {student_class})</h2>'
            f'<p><b>Prediction:</b> At-Risk</p>'
            f'<p><b>Key Areas Identified:</b></p><ul>{"".join(feedback_points)}</ul>'
            f'{ai_html}'  # Add AI improvement strategies
            '</div>'
        )


# --- App Interface ---
st.title("🔮 Aura Predict: Student Success AI")

st.header("Step 1: Tell Us About the Student", divider='blue')
col1, col2 = st.columns(2)
with col1:
    student_name = st.text_input("Student's Full Name", "Alan Turing")
    gender = st.selectbox('Gender', ['Male', 'Female', 'Prefer not to say'])
with col2:
    student_class = st.text_input("Class or Grade", "10")
    parental_education = st.selectbox('Parental Education Level', ['High School', 'College', 'Postgraduate'])

st.header("Step 2: Provide Academic & Lifestyle Vitals", divider='blue')
with st.expander("Click here to enter student's vitals", expanded=True):
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.subheader("Academic Factors")
        hours_studied = st.slider('Hours Studied/Week', 1, 45, 20)
        previous_scores = st.slider('Previous Score', 50, 100, 75)
        attendance = st.slider('Attendance (%)', 60, 100, 80)
        tutoring_sessions = st.slider('Tutoring Sessions', 0, 8, 1)
    with s_col2:
        st.subheader("Personal Habits")
        sleep_hours = st.slider('Sleep Hours/Night', 4, 10, 7)
        physical_activity = st.slider('Activity (hrs/wk)', 0, 6, 3)
        motivation_level = st.selectbox('Motivation Level', ['Low', 'Medium', 'High'])
    with s_col3:
        st.subheader("Environmental Factors")
        parental_involvement = st.selectbox('Parental Involvement', ['Low', 'Medium', 'High'])

st.header("Step 3: Get AI-Powered Analysis", divider='blue')

if st.button('Analyze Student Profile'):
    input_data = {
        'Hours_Studied': hours_studied, 'Attendance': attendance, 'Sleep_Hours': sleep_hours,
        'Previous_Scores': previous_scores, 'Tutoring_Sessions': tutoring_sessions,
        'Physical_Activity': physical_activity, 'Parental_Involvement': parental_involvement,
        'Access_to_Resources': 'Medium', 'Extracurricular_Activities': 'No',
        'Motivation_Level': motivation_level, 'Internet_Access': 'Yes', 'Family_Income': 'Medium',
        'Teacher_Quality': 'Medium', 'School_Type': 'Public', 'Peer_Influence': 'Neutral',
        'Learning_Disabilities': 'No', 'Parental_Education_Level': parental_education,
        'Distance_from_Home': 'Near', 'Gender': gender
    }
    input_df = pd.DataFrame([input_data])

    input_df_processed = pd.get_dummies(input_df)
    input_df_aligned = input_df_processed.reindex(columns=model_columns, fill_value=0)

    with st.spinner(f'Aura is preparing a detailed analysis for {student_name}...'):
        time.sleep(1)
        prediction = model.predict(input_df_aligned)[0]
        feedback_html = generate_feedback(input_df, prediction, student_name, student_class)

    st.markdown(feedback_html, unsafe_allow_html=True)