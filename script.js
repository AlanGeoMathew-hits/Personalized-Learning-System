document.addEventListener('DOMContentLoaded', () => {
  // --- Smooth scroll ---
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', (e) => {
      e.preventDefault();
      document.querySelector(a.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' });
    });
  });

  // --- Theme toggle ---
  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      document.body.classList.toggle('light-mode');
      document.body.classList.toggle('dark-mode');
    });
  }

  // --- Range slider -> output binding ---
  const bindRange = (id, outId, unit = '') => {
    const el = document.getElementById(id);
    const out = document.getElementById(outId);
    if (!el || !out) return;
    const update = () => (out.textContent = el.value + unit);
    el.addEventListener('input', update);
    update();
  };
  bindRange('prev-score', 'out-prev-score');
  bindRange('attendance', 'out-attendance', '%');
  bindRange('hours-studied', 'out-hours-studied');
  bindRange('sleep-hours', 'out-sleep-hours');
  bindRange('physical-activity', 'out-physical-activity');
  bindRange('tutoring', 'out-tutoring');

  // --- Demo logic ---
  const analyzeBtn = document.querySelector('.analyze-button');
  const resultBox = document.getElementById('result-content');

  const showLoading = (btn, loading = true) => {
    const spinner = btn.querySelector('.spinner');
    if (spinner) spinner.hidden = !loading;
    btn.disabled = loading;
  };

  const showPanel = (html) => {
    resultBox.innerHTML = `<div class="panel">${html}</div>`;
  };

  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
      const studentData = {
        name: document.getElementById('name').value,
        grade: document.getElementById('grade').value,
        gender: document.getElementById('gender').value,
        previous_scores: parseInt(document.getElementById('prev-score').value),
        attendance: parseInt(document.getElementById('attendance').value),
        hours_studied: parseInt(document.getElementById('hours-studied').value),
        sleep_hours: parseInt(document.getElementById('sleep-hours').value),
        physical_activity: parseInt(document.getElementById('physical-activity').value),
        tutoring_sessions: parseInt(document.getElementById('tutoring').value),
        motivation_level: document.getElementById('motivation').value,
        parental_involvement: document.getElementById('parental-involvement').value,
        parental_education_level: document.getElementById('parental-education').value,
        teacher_quality: document.getElementById('teacher-quality').value,
        peer_influence: document.getElementById('peer-influence').value,
        extracurricular_activities: document.getElementById('extracurriculars').value,
      };

      if (!studentData.name || !studentData.grade) {
        showPanel(`<p class="placeholder">Please enter the student's name and grade to continue.</p>`);
        return;
      }
      
      showLoading(analyzeBtn, true);
      showPanel(`<p class="placeholder">Preparing a live AI analysis for <strong>${studentData.name}</strong>…</p>`);

      try {
        const res = await fetch('http://127.0.0.1:5000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(studentData)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderResults(data);
      } catch (err) {
        showPanel(`<p class="placeholder"><strong>Error:</strong> Could not connect to the analysis server. Ensure the Python backend is running and not blocked by a firewall.</p>`);
        console.error(err);
      } finally {
        showLoading(analyzeBtn, false);
      }
    });
  }

  function renderResults(data) {
    const statusClass = data.is_at_risk ? 'status-risk' : 'status-safe';
    const statusIcon = data.is_at_risk ? '⚠️' : '✅';
    const statusText = data.is_at_risk ? 'At-Risk' : 'Not At-Risk';

    const converter = new showdown.Converter({ghCompatibleHeaderId: true, simpleLineBreaks: true});
    const recommendationsHtml = converter.makeHtml(data.recommendations);
    
    showPanel(`
      <h4 class="${statusClass}">${statusIcon} Prediction: ${statusText}</h4>
      <p><strong>Model Confidence:</strong> ${(data.confidence * 100).toFixed(0)}%</p>
      <hr>
      ${recommendationsHtml}
    `);
  }
});