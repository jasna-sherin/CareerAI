import pandas as pd
import random

"""
Sample Data Generator for Career Path Recommendation System
Generates realistic student data for training the ML model
"""

# Define possible values for each field
GENDERS = ['Male', 'Female']
LOCATIONS = ['Mumbai', 'Bangalore', 'Delhi', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata', 'Ahmedabad']
QUALIFICATIONS = ['B.Tech', 'M.Tech', 'BCA', 'MCA', 'B.Sc', 'M.Sc', 'MBA']
STREAMS = ['CSE', 'IT', 'ECE', 'EEE', 'Mechanical', 'Civil', 'Mathematics', 'Physics']
ACADEMIC_LEVELS = ['Student', 'Graduate']
WORK_STYLES = ['Remote', 'Office', 'Hybrid']
WORK_TYPES = ['Technical', 'Managerial', 'Creative', 'Analytical', 'Research']
RELOCATE = ['Yes', 'No', 'Maybe']

# Technical skills by domain
TECH_SKILLS = {
    'Data Science': ['Python', 'Machine Learning', 'Statistics', 'SQL', 'Data Visualization', 'Pandas', 'NumPy', 'Scikit-learn'],
    'Software Development': ['Java', 'Python', 'C++', 'Data Structures', 'Algorithms', 'Git', 'API Development'],
    'Web Development': ['JavaScript', 'HTML', 'CSS', 'React', 'Node.js', 'Angular', 'Vue.js', 'MongoDB'],
    'Mobile Development': ['Swift', 'Kotlin', 'React Native', 'Flutter', 'Firebase', 'API Integration'],
    'AI/ML': ['Python', 'TensorFlow', 'PyTorch', 'Deep Learning', 'Neural Networks', 'Computer Vision', 'NLP'],
    'Testing': ['Selenium', 'JUnit', 'TestNG', 'Automation', 'API Testing', 'Performance Testing'],
    'DevOps': ['Docker', 'Kubernetes', 'AWS', 'CI/CD', 'Jenkins', 'Linux', 'Scripting'],
    'Cloud': ['AWS', 'Azure', 'GCP', 'Terraform', 'CloudFormation', 'Serverless']
}

SOFT_SKILLS = [
    'Communication', 'Leadership', 'Teamwork', 'Problem Solving', 'Time Management',
    'Critical Thinking', 'Adaptability', 'Creativity', 'Analytical Skills', 'Attention to Detail'
]

LANGUAGES = ['English', 'Hindi', 'Telugu', 'Tamil', 'Kannada', 'Malayalam', 'Marathi', 'Bengali']

INTERESTS = ['Data Science', 'Software Development', 'Web Development', 'Mobile Development', 
             'AI/ML', 'Cloud Computing', 'Cybersecurity', 'Testing', 'Management', 'Research']

CERTIFICATIONS = [
    'AWS Certified', 'Google Analytics', 'Microsoft Azure', 'Coursera ML', 'Oracle Certified',
    'Scrum Master', 'PMP', 'CompTIA', 'Cisco CCNA', ''
]

EXPERIENCES = [
    'Intern at Tech Company', 'Part-time Developer', 'Freelance Projects', 
    'College Project Lead', 'Hackathon Winner', 'Open Source Contributor', ''
]

# Career paths with their typical characteristics
CAREER_PROFILES = {
    'Data Scientist': {
        'avg_grade': (78, 95),
        'interests': ['Data Science', 'AI/ML', 'Research'],
        'qualifications': ['B.Tech', 'M.Tech', 'M.Sc'],
        'streams': ['CSE', 'IT', 'Mathematics', 'Statistics'],
        'work_type': ['Technical', 'Analytical', 'Research']
    },
    'Software Developer': {
        'avg_grade': (70, 92),
        'interests': ['Software Development', 'Web Development'],
        'qualifications': ['B.Tech', 'BCA', 'MCA'],
        'streams': ['CSE', 'IT', 'ECE'],
        'work_type': ['Technical']
    },
    'Web Developer': {
        'avg_grade': (65, 88),
        'interests': ['Web Development', 'Software Development'],
        'qualifications': ['B.Tech', 'BCA', 'MCA'],
        'streams': ['CSE', 'IT'],
        'work_type': ['Technical', 'Creative']
    },
    'Mobile Developer': {
        'avg_grade': (68, 90),
        'interests': ['Mobile Development', 'Software Development'],
        'qualifications': ['B.Tech', 'BCA', 'MCA'],
        'streams': ['CSE', 'IT'],
        'work_type': ['Technical']
    },
    'Data Analyst': {
        'avg_grade': (70, 90),
        'interests': ['Data Science', 'Research'],
        'qualifications': ['B.Tech', 'B.Sc', 'MBA'],
        'streams': ['CSE', 'IT', 'Mathematics'],
        'work_type': ['Analytical', 'Technical']
    },
    'QA Engineer': {
        'avg_grade': (65, 85),
        'interests': ['Testing', 'Software Development'],
        'qualifications': ['B.Tech', 'BCA'],
        'streams': ['CSE', 'IT', 'ECE'],
        'work_type': ['Technical']
    },
    'Product Manager': {
        'avg_grade': (75, 92),
        'interests': ['Management', 'Software Development'],
        'qualifications': ['B.Tech', 'MBA'],
        'streams': ['CSE', 'IT'],
        'work_type': ['Managerial', 'Analytical']
    }
}

def generate_student_record(career_path):
    """Generate a realistic student record for a given career path"""
    
    profile = CAREER_PROFILES[career_path]
    
    # Age based on qualification
    qualification = random.choice(profile['qualifications'])
    if qualification in ['B.Tech', 'BCA', 'B.Sc']:
        age = random.randint(20, 23)
    else:
        age = random.randint(23, 26)
    
    # Grade from career-specific range
    grade = round(random.uniform(*profile['avg_grade']), 1)
    
    # Interest aligned with career
    interest = random.choice(profile['interests'])
    
    # Get relevant technical skills
    tech_skills_list = TECH_SKILLS.get(interest, ['Python', 'Java'])
    num_skills = random.randint(2, min(6, len(tech_skills_list)))
    technical_skills = ', '.join(random.sample(tech_skills_list, num_skills))
    
    # Soft skills
    num_soft = random.randint(2, 4)
    soft_skills = ', '.join(random.sample(SOFT_SKILLS, num_soft))
    
    # Languages
    num_langs = random.randint(2, 3)
    languages = ', '.join(['English'] + random.sample([l for l in LANGUAGES if l != 'English'], num_langs - 1))
    
    # Certifications (70% chance)
    certifications = random.choice(CERTIFICATIONS) if random.random() < 0.7 else ''
    
    # Experience (60% chance)
    experience = random.choice(EXPERIENCES) if random.random() < 0.6 else ''
    
    record = {
        'Age': age,
        'Gender': random.choice(GENDERS),
        'Location': random.choice(LOCATIONS),
        'Highest_qualification': qualification,
        'Stream': random.choice(profile['streams']),
        'Current_Academic_Level': random.choice(ACADEMIC_LEVELS),
        'Grade_CGPA_Percentage': grade,
        'Technical_Skills': technical_skills,
        'Soft_Skills': soft_skills,
        'Languages_Known': languages,
        'Certifications': certifications,
        'Fields_of_Interest': interest,
        'Preferred_Work_Style': random.choice(WORK_STYLES),
        'Work_Type_Interest': random.choice(profile['work_type']),
        'Past_Jobs_Internships': experience,
        'Achievements': 'Academic Excellence' if grade > 85 else '',
        'Skills_Gained': '',
        'Willing_to_Relocate': random.choice(RELOCATE),
        'Suggested_Career_Path': career_path
    }
    
    return record

def generate_dataset(num_records=500):
    """Generate a balanced dataset with multiple career paths"""
    
    print(f"Generating {num_records} student records...")
    
    careers = list(CAREER_PROFILES.keys())
    records_per_career = num_records // len(careers)
    
    data = []
    
    for career in careers:
        print(f"Generating {records_per_career} records for {career}...")
        for _ in range(records_per_career):
            data.append(generate_student_record(career))
    
    # Add remaining records to balance exactly
    remaining = num_records - len(data)
    for _ in range(remaining):
        career = random.choice(careers)
        data.append(generate_student_record(career))
    
    # Shuffle the data
    random.shuffle(data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('career_data.csv', index=False)
    
    print(f"\n✅ Successfully generated {len(df)} records!")
    print(f"📁 Saved to: career_data.csv")
    print(f"\nCareer Distribution:")
    print(df['Suggested_Career_Path'].value_counts())
    print(f"\nAverage Grades by Career:")
    print(df.groupby('Suggested_Career_Path')['Grade_CGPA_Percentage'].mean().round(2))
    
    return df

if __name__ == "__main__":
    # Generate dataset
    # You can change the number of records here
    num_records = 500
    
    print("=" * 60)
    print("Career Path Recommendation System - Data Generator")
    print("=" * 60)
    print()
    
    df = generate_dataset(num_records)
    
    print("\n" + "=" * 60)
    print("Sample Records:")
    print("=" * 60)
    print(df.head(3))
    
    print("\n✨ Data generation complete!")
    print("📝 Next step: Run 'python train_model.py' to train the model")
