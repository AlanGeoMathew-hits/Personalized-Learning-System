# 🎓 Personalized Learning System

An intelligent, adaptive learning platform designed to deliver customized educational paths, track student progress, and recommend tailored resources based on individual performance and learning analytics.

---

## 🌟 Features

- **Adaptive Curriculum Generation:** Dynamically adjusts learning paths and difficulty levels based on user quiz scores and engagement metrics.
- **Smart Recommendations:** Uses machine learning algorithms and heuristic models to suggest relevant topics, practice problems, and reading materials.
- **Progress Tracking & Analytics:** Comprehensive dashboards displaying learning velocity, topic mastery percentages, and historical performance.
- **Interactive Assessments:** Built-in quizzes and coding/aptitude problem sets with instant feedback.
- **Modular Architecture:** Scalable backend API coupled with a responsive frontend interface.

---

## 🛠️ Tech Stack

- **Backend:** Python (Flask / FastAPI)
- **Machine Learning / Intelligence Engine:** Scikit-learn, NumPy, Pandas (for recommendation and profiling models)
- **Frontend:** HTML5, CSS3, JavaScript / React
- **Database:** PostgreSQL / SQLite / JSON-based storage
- **Version Control:** Git & GitHub

---

## 📂 Project Structure

```text
Personalized-Learning-System/
│
├── backend/               # API endpoints, business logic, and server setup
│   ├── app.py             # Main application entry point
│   ├── models/            # Database models and ML recommendation scripts
│   └── routes/            # API route handlers (users, courses, assessments)
│
├── frontend/              # User interface components and assets
│   ├── index.html         # Main landing/dashboard page
│   ├── css/               # Stylesheets
│   └── js/                # Client-side logic and API integration
│
├── data/                  # Sample datasets, question banks, and learning modules
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally on your machine.

### Prerequisites

Ensure you have the following installed:
- Python (v3.8 or higher)
- pip (Python package manager)
- Git

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AlanGeoMathew-hits/Personalized-Learning-System.git
   cd Personalized-Learning-System
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory (referencing `.env.example`) and add your configuration settings:
   ```env
   FLASK_ENV=development
   PORT=5000
   SECRET_KEY=your_secret_key_here
   ```

5. **Run the Application:**
   ```bash
   python backend/app.py
   ```

6. **Access the Application:**
   Open your browser and navigate to `http://localhost:5000` (or your configured port).

---

## 📊 How Personalization Works

1. **Profiling:** The system evaluates the user's initial baseline knowledge through a diagnostic test.
2. **Analysis:** Performance logs are analyzed via recommendation algorithms to identify knowledge gaps and strengths.
3. **Adaptation:** The learning engine updates the user's curriculum queue in real-time, prioritizing concepts that require reinforcement.

---

## 🔮 Future Roadmap

- [ ] Integrate advanced LLM-based tutoring assistants for real-time query resolution.
- [ ] Implement user authentication with OAuth2.
- [ ] Add support for multi-format learning resources (video tutorials, interactive code sandboxes, and PDF notes).
- [ ] Deploy the platform to cloud infrastructure (Render/AWS).

---

## 👤 Author

**Alan Geo Mathew**  
- GitHub: [@AlanGeoMathew-hits](https://github.com/AlanGeoMathew-hits)

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
