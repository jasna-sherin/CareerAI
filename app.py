from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json
import random

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-this-in-production-please')

# ── Model loading (graceful fallback) ─────────────────────────────────────────
model = None
label_encoders = None
feature_names = None
MODEL_AVAILABLE = False

try:
    import pickle
    with open('models/career_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    MODEL_AVAILABLE = True
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️  Model not found ({e}). Running in demo mode with mock predictions.")

# ── Data storage ───────────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
HISTORY_FILE = 'data/prediction_history.json'
FEEDBACK_FILE = 'data/user_feedback.json'

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def append_json(path, entry, max_entries=200):
    data = load_json(path)
    data.append(entry)
    data = data[-max_entries:]
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

# ── Career database ────────────────────────────────────────────────────────────
CAREER_INFO = {
    'Data Scientist': {
        'icon': '🔬',
        'description': 'Analyze complex datasets to extract actionable insights and build predictive models that drive strategic business decisions.',
        'skills_required': ['Python', 'Machine Learning', 'Statistics', 'SQL', 'Data Visualization', 'TensorFlow'],
        'avg_salary': '$95,000 – $150,000',
        'growth_rate': '36% growth (Much faster than average)',
        'education': "Bachelor's or Master's in Computer Science, Statistics, or a related field",
        'companies': ['Google', 'Amazon', 'Microsoft', 'Meta', 'Netflix', 'Spotify'],
        'related_careers': ['Data Analyst', 'ML Engineer', 'AI Researcher', 'Business Intelligence Analyst']
    },
    'Software Developer': {
        'icon': '💻',
        'description': 'Design, develop, and maintain software systems that power modern applications across web, desktop, and enterprise environments.',
        'skills_required': ['Programming', 'Problem Solving', 'Data Structures', 'Algorithms', 'Version Control', 'Testing'],
        'avg_salary': '$80,000 – $130,000',
        'growth_rate': '25% growth (Much faster than average)',
        'education': "Bachelor's in Computer Science or related field",
        'companies': ['Microsoft', 'Google', 'Apple', 'Amazon', 'IBM', 'Atlassian'],
        'related_careers': ['Full Stack Developer', 'Backend Developer', 'DevOps Engineer', 'Systems Architect']
    },
    'Web Developer': {
        'icon': '🌐',
        'description': 'Build visually engaging and performant websites and web applications that deliver excellent user experiences across all devices.',
        'skills_required': ['HTML/CSS', 'JavaScript', 'React/Angular', 'Node.js', 'REST APIs', 'Responsive Design'],
        'avg_salary': '$60,000 – $110,000',
        'growth_rate': '23% growth (Much faster than average)',
        'education': "Bachelor's in Computer Science or bootcamp certification",
        'companies': ['Shopify', 'WordPress', 'Adobe', 'Squarespace', 'Cloudflare', 'Vercel'],
        'related_careers': ['Frontend Developer', 'UI/UX Designer', 'Full Stack Developer', 'JAMstack Developer']
    },
    'Mobile Developer': {
        'icon': '📱',
        'description': 'Create intuitive and high-performance mobile applications for iOS and Android that users love.',
        'skills_required': ['Swift/Kotlin', 'React Native', 'Mobile UI Design', 'API Integration', 'App Store Deployment'],
        'avg_salary': '$75,000 – $125,000',
        'growth_rate': '24% growth (Much faster than average)',
        'education': "Bachelor's in Computer Science or Mobile Development",
        'companies': ['Apple', 'Google', 'Uber', 'Spotify', 'Instagram', 'Airbnb'],
        'related_careers': ['iOS Developer', 'Android Developer', 'Flutter Developer', 'Game Developer']
    },
    'Data Analyst': {
        'icon': '📊',
        'description': 'Collect, process, and interpret data to provide clear insights that guide business strategy and operational improvements.',
        'skills_required': ['Excel', 'SQL', 'Python/R', 'Tableau/Power BI', 'Business Intelligence', 'Statistical Analysis'],
        'avg_salary': '$65,000 – $95,000',
        'growth_rate': '25% growth (Much faster than average)',
        'education': "Bachelor's in Statistics, Mathematics, or Business Analytics",
        'companies': ['McKinsey', 'Deloitte', 'Accenture', 'IBM', 'Oracle', 'Palantir'],
        'related_careers': ['Business Analyst', 'Data Scientist', 'Market Research Analyst', 'BI Developer']
    },
    'QA Engineer': {
        'icon': '🔍',
        'description': 'Ensure software quality through rigorous testing strategies, automated test suites, and continuous quality improvement practices.',
        'skills_required': ['Selenium', 'Test Automation', 'JIRA', 'CI/CD', 'API Testing', 'Performance Testing'],
        'avg_salary': '$60,000 – $100,000',
        'growth_rate': '22% growth (Faster than average)',
        'education': "Bachelor's in Computer Science or related field",
        'companies': ['Amazon', 'Microsoft', 'Adobe', 'Salesforce', 'Oracle', 'ServiceNow'],
        'related_careers': ['Test Automation Engineer', 'Performance Tester', 'DevOps Engineer', 'SDET']
    },
    'Product Manager': {
        'icon': '🚀',
        'description': 'Define product vision and roadmap, align cross-functional teams, and translate user needs into products that create real impact.',
        'skills_required': ['Leadership', 'Market Analysis', 'Communication', 'Agile/Scrum', 'User Research', 'Roadmapping'],
        'avg_salary': '$90,000 – $150,000',
        'growth_rate': '20% growth (Faster than average)',
        'education': "Bachelor's in Business, Engineering, or MBA preferred",
        'companies': ['Google', 'Amazon', 'Microsoft', 'Meta', 'Apple', 'Stripe'],
        'related_careers': ['Program Manager', 'Business Analyst', 'Scrum Master', 'Growth Manager']
    },
    'ML Engineer': {
        'icon': '🤖',
        'description': 'Build and deploy machine learning systems at scale, bridging the gap between research models and production applications.',
        'skills_required': ['Python', 'PyTorch/TensorFlow', 'MLOps', 'Docker/Kubernetes', 'Cloud Platforms', 'Feature Engineering'],
        'avg_salary': '$110,000 – $170,000',
        'growth_rate': '40% growth (Exceptionally fast)',
        'education': "Master's or PhD in CS, ML, or related field preferred",
        'companies': ['OpenAI', 'DeepMind', 'Google AI', 'NVIDIA', 'Anthropic', 'Scale AI'],
        'related_careers': ['Data Scientist', 'AI Researcher', 'Deep Learning Engineer', 'NLP Engineer']
    },
    'Cloud Engineer': {
        'icon': '☁️',
        'description': 'Design, implement, and manage scalable cloud infrastructure that powers modern applications with high availability and security.',
        'skills_required': ['AWS/Azure/GCP', 'Terraform', 'Docker', 'Kubernetes', 'Networking', 'Security'],
        'avg_salary': '$95,000 – $145,000',
        'growth_rate': '32% growth (Much faster than average)',
        'education': "Bachelor's in Computer Science, Cloud certifications highly valued",
        'companies': ['AWS', 'Microsoft Azure', 'Google Cloud', 'Cloudflare', 'HashiCorp', 'Datadog'],
        'related_careers': ['DevOps Engineer', 'Site Reliability Engineer', 'Platform Engineer', 'Solutions Architect']
    },
    'Cybersecurity Analyst': {
        'icon': '🛡️',
        'description': 'Protect organizations from cyber threats through threat analysis, incident response, security architecture, and proactive defense.',
        'skills_required': ['Network Security', 'Penetration Testing', 'SIEM Tools', 'Incident Response', 'Cryptography', 'Compliance'],
        'avg_salary': '$85,000 – $140,000',
        'growth_rate': '33% growth (Much faster than average)',
        'education': "Bachelor's in Cybersecurity, Computer Science, or related field",
        'companies': ['CrowdStrike', 'Palo Alto Networks', 'Cisco', 'IBM Security', 'FireEye', 'Splunk'],
        'related_careers': ['Security Engineer', 'Penetration Tester', 'CISO', 'Threat Intelligence Analyst']
    },
    'Deep Learning Engineer': {
        'icon': '🧠',
        'description': 'Research and implement advanced neural network architectures for computer vision, NLP, and other cutting-edge AI applications.',
        'skills_required': ['PyTorch', 'TensorFlow', 'Computer Vision', 'NLP', 'CUDA Programming', 'Research Papers'],
        'avg_salary': '$120,000 – $200,000',
        'growth_rate': '45% growth (Exceptionally fast)',
        'education': "Master's or PhD in Machine Learning, Computer Science, or related field",
        'companies': ['OpenAI', 'DeepMind', 'NVIDIA', 'Google Brain', 'Meta AI', 'Tesla AI'],
        'related_careers': ['ML Engineer', 'AI Researcher', 'Computer Vision Engineer', 'NLP Engineer']
    },
    'Business Analyst': {
        'icon': '📈',
        'description': 'Bridge business needs and technical solutions by analyzing processes, identifying opportunities, and driving data-informed decisions.',
        'skills_required': ['Requirements Analysis', 'SQL', 'Process Modeling', 'Stakeholder Management', 'Excel', 'Agile'],
        'avg_salary': '$70,000 – $110,000',
        'growth_rate': '18% growth (Faster than average)',
        'education': "Bachelor's in Business, IT, or related field; MBA a plus",
        'companies': ['Deloitte', 'Accenture', 'McKinsey', 'PwC', 'KPMG', 'Cognizant'],
        'related_careers': ['Data Analyst', 'Product Manager', 'Systems Analyst', 'Operations Analyst']
    },
    'Software Engineer': {
        'icon': '⚙️',
        'description': 'Apply engineering principles to design robust, scalable software systems with a focus on architecture, performance, and maintainability.',
        'skills_required': ['System Design', 'Algorithms', 'OOP', 'Distributed Systems', 'Testing', 'Code Review'],
        'avg_salary': '$90,000 – $145,000',
        'growth_rate': '25% growth (Much faster than average)',
        'education': "Bachelor's in Computer Science or Engineering",
        'companies': ['Google', 'Apple', 'Amazon', 'Meta', 'Netflix', 'Stripe'],
        'related_careers': ['Software Developer', 'Systems Architect', 'Platform Engineer', 'Staff Engineer']
    }
}

ALL_CAREERS = list(CAREER_INFO.keys())

def get_career_info(career_path):
    return CAREER_INFO.get(career_path, {
        'icon': '🎯',
        'description': 'An exciting career opportunity in the technology sector.',
        'skills_required': ['Technical Skills', 'Problem Solving', 'Communication', 'Teamwork'],
        'avg_salary': 'Competitive salary',
        'growth_rate': 'Growing field',
        'education': 'Relevant degree or professional certification',
        'companies': ['Various leading tech companies'],
        'related_careers': []
    })

# ── Mock prediction (demo mode) ────────────────────────────────────────────────
def mock_predict(data):
    """Generate a plausible mock prediction when model is unavailable."""
    interest = data.get('interest', '').lower()
    interest_map = {
        'data science': 'Data Scientist',
        'ai/ml': 'ML Engineer',
        'software development': 'Software Developer',
        'web development': 'Web Developer',
        'mobile development': 'Mobile Developer',
        'cloud computing': 'Cloud Engineer',
        'cybersecurity': 'Cybersecurity Analyst',
        'testing': 'QA Engineer',
        'management': 'Product Manager',
        'research': 'Deep Learning Engineer',
    }
    primary = interest_map.get(interest, random.choice(ALL_CAREERS))
    others = [c for c in ALL_CAREERS if c != primary]
    random.shuffle(others)
    top3 = [primary] + others[:2]

    # Generate plausible probabilities
    probs = sorted([random.uniform(0.15, 0.45), random.uniform(0.1, 0.2), random.uniform(0.05, 0.15)], reverse=True)
    top_careers = [{'career': c, 'probability': round(p * 100, 2)} for c, p in zip(top3, probs)]
    confidence = top_careers[0]['probability']
    return primary, confidence, top_careers

# ── Input preprocessing ────────────────────────────────────────────────────────
def preprocess_input(data):
    if not MODEL_AVAILABLE:
        return None
    features = {}
    features['Age'] = int(data.get('age', 20))
    features['Grade_CGPA_Percentage'] = float(data.get('grade', 75.0))
    tech = data.get('technical_skills', '').strip()
    features['Technical_Skills_Count'] = len([s for s in tech.split(',') if s.strip()]) if tech else 0
    soft = data.get('soft_skills', '').strip()
    features['Soft_Skills_Count'] = len([s for s in soft.split(',') if s.strip()]) if soft else 0
    langs = data.get('languages', '').strip()
    features['Languages_Count'] = len([s for s in langs.split(',') if s.strip()]) if langs else 0
    features['Has_Certifications'] = 1 if data.get('certifications', '').strip() else 0
    features['Has_Experience'] = 1 if data.get('experience', '').strip() else 0
    categorical_map = {
        'Gender': data.get('gender', 'Male'),
        'Location': data.get('location', 'Unknown'),
        'Highest_qualification': data.get('qualification', 'B.Tech'),
        'Stream': data.get('stream', 'CSE'),
        'Current_Academic_Level': data.get('academic_level', 'Student'),
        'Fields_of_Interest': data.get('interest', 'Technology'),
        'Preferred_Work_Style': data.get('work_style', 'Office'),
        'Work_Type_Interest': data.get('work_type', 'Technical'),
        'Willing_to_Relocate': data.get('relocate', 'Yes')
    }
    for feature, value in categorical_map.items():
        encoder = label_encoders[feature]
        try:
            enc_val = encoder.transform([str(value)])[0]
        except ValueError:
            enc_val = 0
        features[feature + '_encoded'] = enc_val
    feature_array = [features[fname] for fname in feature_names]
    return np.array(feature_array).reshape(1, -1)

def run_prediction(data):
    """Unified prediction — uses real model or falls back to mock."""
    if MODEL_AVAILABLE:
        input_features = preprocess_input(data)
        prediction = model.predict(input_features)[0]
        prediction_proba = model.predict_proba(input_features)[0]
        target_encoder = label_encoders['target']
        career_path = target_encoder.inverse_transform([prediction])[0]
        top_indices = np.argsort(prediction_proba)[-3:][::-1]
        top_careers = []
        for idx in top_indices:
            career = target_encoder.inverse_transform([idx])[0]
            top_careers.append({'career': career, 'probability': round(prediction_proba[idx] * 100, 2)})
        confidence = round(prediction_proba[prediction] * 100, 2)
    else:
        career_path, confidence, top_careers = mock_predict(data)
    return career_path, confidence, top_careers

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html', demo_mode=not MODEL_AVAILABLE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        career_path, confidence, top_careers = run_prediction(data)
        career_details = get_career_info(career_path)
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'career_path': career_path,
            'confidence': confidence,
            'student_info': {
                'age': data.get('age'),
                'qualification': data.get('qualification'),
                'grade': data.get('grade'),
                'interest': data.get('interest')
            }
        }
        append_json(HISTORY_FILE, history_entry)
        session['last_prediction'] = {
            'career_path': career_path,
            'confidence': confidence,
            'top_careers': top_careers,
            'student_data': data
        }
        return render_template('result.html',
                               career_path=career_path,
                               confidence=confidence,
                               top_careers=top_careers,
                               career_details=career_details,
                               student_data=data,
                               all_careers=ALL_CAREERS,
                               demo_mode=not MODEL_AVAILABLE)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/api/compare', methods=['POST'])
def api_compare():
    try:
        data = request.get_json()
        s1, s2 = data.get('student1', {}), data.get('student2', {})
        c1, conf1, top1 = run_prediction(s1)
        c2, conf2, top2 = run_prediction(s2)
        return jsonify({
            'success': True,
            'student1': {'career': c1, 'confidence': conf1, 'top': top1, 'info': get_career_info(c1)},
            'student2': {'career': c2, 'confidence': conf2, 'top': top2, 'info': get_career_info(c2)}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/history')
def history():
    history_data = load_json(HISTORY_FILE)
    return render_template('history.html', history=list(reversed(history_data)))

@app.route('/statistics')
def statistics():
    history_data = load_json(HISTORY_FILE)
    total = len(history_data)
    career_counts = {}
    for entry in history_data:
        c = entry.get('career_path', 'Unknown')
        career_counts[c] = career_counts.get(c, 0) + 1
    avg_confidence = round(sum(e.get('confidence', 0) for e in history_data) / total, 2) if total > 0 else 0
    sorted_careers = sorted(career_counts.items(), key=lambda x: x[1], reverse=True)
    max_count = sorted_careers[0][1] if sorted_careers else 1
    stats = {
        'total_predictions': total,
        'career_distribution': dict(sorted_careers),
        'max_count': max_count,
        'avg_confidence': avg_confidence,
        'recent_predictions': list(reversed(history_data))[:10]
    }
    return render_template('statistics.html', stats=stats)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rating': request.form.get('rating'),
            'prediction_accurate': request.form.get('accurate'),
            'comments': request.form.get('comments'),
            'suggested_career': request.form.get('suggested_career')
        }
        append_json(FEEDBACK_FILE, entry)
        return redirect(url_for('feedback_success'))
    return render_template('feedback.html')

@app.route('/feedback/success')
def feedback_success():
    return render_template('feedback_success.html')

@app.route('/career-guide/<path:career_name>')
def career_guide(career_name):
    career_info = get_career_info(career_name)
    all_careers = list(CAREER_INFO.keys())
    return render_template('career_guide.html',
                           career_name=career_name,
                           career_info=career_info,
                           all_careers=all_careers)

@app.route('/batch-predict', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('error.html', error='No file uploaded. Please select a CSV file.')
        file = request.files['file']
        if not file or file.filename == '':
            return render_template('error.html', error='No file selected.')
        if not file.filename.lower().endswith('.csv'):
            return render_template('error.html', error='Only CSV files are supported.')
        try:
            df = pd.read_csv(file)
            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]
            results = []
            for idx, row in df.iterrows():
                data = {k: str(v) for k, v in row.to_dict().items() if pd.notna(v)}
                career, confidence, top = run_prediction(data)
                results.append({
                    'student_id': idx + 1,
                    'name': data.get('name', f'Student {idx + 1}'),
                    'predicted_career': career,
                    'confidence': confidence,
                    'top_careers': top
                })
                if idx >= 999:
                    break
            return render_template('batch_results.html', results=results)
        except Exception as e:
            return render_template('error.html', error=f'Error processing file: {str(e)}')
    return render_template('batch_predict.html')

@app.route('/skills-gap-analysis', methods=['POST'])
def skills_gap_analysis():
    try:
        data = request.form.to_dict()
        desired_career = data.get('desired_career', '')
        current_skills_raw = data.get('technical_skills', '')
        current_skills = [s.strip() for s in current_skills_raw.split(',') if s.strip()]
        career_info = get_career_info(desired_career)
        required_skills = career_info.get('skills_required', [])
        current_lower = [s.lower() for s in current_skills]
        matching = [s for s in required_skills if s.lower() in current_lower]
        missing = [s for s in required_skills if s.lower() not in current_lower]
        pct = round(len(matching) / len(required_skills) * 100, 2) if required_skills else 0
        analysis = {
            'desired_career': desired_career,
            'career_icon': career_info.get('icon', '🎯'),
            'required_skills': required_skills,
            'your_skills': current_skills,
            'matching_skills': matching,
            'missing_skills': missing,
            'match_percentage': pct
        }
        return render_template('skills_gap.html', analysis=analysis)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json() or {}
        career_path, confidence, top_careers = run_prediction(data)
        return jsonify({
            'success': True,
            'predicted_career': career_path,
            'confidence': confidence,
            'top_careers': top_careers,
            'career_details': get_career_info(career_path)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error='Page not found (404).'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error='Internal server error. Please try again.'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
