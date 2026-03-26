# CareerAI — Career Path Recommendation System

A Flask web application that recommends career paths for students using a trained ML model (with graceful demo fallback if the model is unavailable).

## Project Structure

```
project/
├── app.py                  # Flask application (main backend)
├── requirements.txt
├── static/
│   └── style.css           # Unified stylesheet for all pages
├── templates/              # Jinja2 HTML templates
│   ├── index.html          # Home / prediction form
│   ├── result.html         # Single prediction result
│   ├── compare.html        # Side-by-side student comparison
│   ├── batch_predict.html  # CSV batch upload form
│   ├── batch_results.html  # Batch prediction results table
│   ├── career_guide.html   # Detailed career guide page
│   ├── statistics.html     # Analytics dashboard
│   ├── history.html        # Full prediction history
│   ├── feedback.html       # User feedback form
│   ├── feedback_success.html
│   ├── skills_gap.html     # Skills gap analysis
│   └── error.html          # Error page
├── data/
│   ├── prediction_history.json
│   └── user_feedback.json
└── models/                 # Place your trained model files here
    ├── career_model.pkl
    ├── label_encoders.pkl
    └── feature_names.pkl
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Place your model files in models/
mkdir models
# copy career_model.pkl, label_encoders.pkl, feature_names.pkl into models/

# 3. Run the app
python app.py
```

App runs at **http://localhost:5000**

## Demo Mode

If `models/` files are not found, the app runs in **demo mode** with simulated predictions. A warning banner is shown in the UI. Everything else works normally.

## Fixes Applied

- **Graceful model loading** — app starts even without model files
- **Fixed statistics crash** — no more division-by-zero when history is empty
- **Fixed emoji encoding** — all garbled `ðŸŽ"` characters replaced with proper Unicode
- **Fixed feedback redirect** — uses `redirect(url_for('feedback_success'))`
- **Added 404/500 error handlers**
- **Expanded career database** — 13 career profiles with icons, salaries, companies
- **Better batch CSV handling** — normalizes column names, handles missing fields
- **Unified navigation** — consistent top nav on every page
- **Fully mobile responsive** — CSS Grid-based layout
