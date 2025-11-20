# -*- coding: utf-8 -*-
# ============================================================================
# TOUCHLESS SATISFACTION SURVEY - ENHANCED EDUCATIONAL VERSION
# Features:
# - Admin panel with password (no default message shown to respondents)
# - Multiple data cleaning strategies with educational notes
# - Multiple statistical methods with explanations
# - Multiple ML models with learning descriptions
# - Built-in SQLite database
# - Teachable Machine integration
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime
import json
import io
import base64

# Analysis libraries
try:
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.impute import KNNImputer, SimpleImputer
    import matplotlib.pyplot as plt
    import seaborn as sns
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Touchless Survey System",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default admin password (NOT shown to users)
DEFAULT_ADMIN_PASSWORD = "admin123"

# Database file
DB_FILE = "survey_responses.db"

# Survey questions
SURVEY_QUESTIONS = [
    "How satisfied are you with the workshop content?",
    "How satisfied are you with the instructor's teaching?",
    "How satisfied are you with the workshop materials?",
    "How satisfied are you with the hands-on activities?",
    "How satisfied are you with the overall workshop experience?"
]

# Gesture mapping
GESTURE_MAP = {
    'thumbs_up': {'label': 'Satisfied', 'score': 4, 'emoji': '👍'},
    'heart_sign': {'label': 'Very Satisfied', 'score': 5, 'emoji': '❤️'},
    'thumbs_down': {'label': 'Unsatisfied', 'score': 2, 'emoji': '👎'},
    'waving_finger': {'label': 'Very Unsatisfied', 'score': 1, 'emoji': '☝️'},
    'closed_fist': {'label': 'No Answer', 'score': None, 'emoji': '✊'}
}

# ============================================================================
# EDUCATIONAL CONTENT - IMPUTATION STRATEGIES
# ============================================================================

IMPUTATION_STRATEGIES = {
    'mean': {
        'name': 'Mean Imputation',
        'description': 'Replaces missing values with the mean (average) of the column.',
        'when_to_use': 'Best for normally distributed data without outliers.',
        'pros': 'Simple, fast, preserves the mean of the dataset.',
        'cons': 'Reduces variance, ignores relationships between variables.',
        'example': 'If scores are [1, 2, ?, 4, 5], missing value becomes 3.0'
    },
    'median': {
        'name': 'Median Imputation',
        'description': 'Replaces missing values with the median (middle value) of the column.',
        'when_to_use': 'Best when data has outliers or is skewed.',
        'pros': 'Robust to outliers, preserves central tendency.',
        'cons': 'Reduces variance, ignores relationships between variables.',
        'example': 'If scores are [1, 2, ?, 4, 100], missing value becomes 3.0 (not affected by 100)'
    },
    'mode': {
        'name': 'Mode Imputation',
        'description': 'Replaces missing values with the most frequent value.',
        'when_to_use': 'Best for categorical data or discrete scores.',
        'pros': 'Preserves the most common response pattern.',
        'cons': 'Can overrepresent the mode, not suitable for continuous data.',
        'example': 'If scores are [5, 4, ?, 5, 5, 3], missing value becomes 5'
    },
    'forward_fill': {
        'name': 'Forward Fill (FFill)',
        'description': 'Fills missing values with the previous valid observation.',
        'when_to_use': 'Best for time-series data or sequential responses.',
        'pros': 'Maintains continuity, no calculation needed.',
        'cons': 'Assumes pattern continues, not suitable for random missing data.',
        'example': 'If sequence is [3, 4, ?, ?, 5], missing values become [3, 4, 4, 4, 5]'
    },
    'backward_fill': {
        'name': 'Backward Fill (BFill)',
        'description': 'Fills missing values with the next valid observation.',
        'when_to_use': 'Best for time-series when future value is more relevant.',
        'pros': 'Uses future information, maintains continuity.',
        'cons': 'Assumes reverse pattern, not suitable for random missing data.',
        'example': 'If sequence is [3, ?, ?, 5, 4], missing values become [3, 5, 5, 5, 4]'
    },
    'interpolate': {
        'name': 'Linear Interpolation',
        'description': 'Estimates missing values by drawing a line between known values.',
        'when_to_use': 'Best for continuous data with smooth trends.',
        'pros': 'Creates smooth transitions, uses surrounding context.',
        'cons': 'Assumes linear relationship, needs values on both sides.',
        'example': 'If scores are [2, ?, ?, 6], missing values become [2, 3.33, 4.67, 6]'
    },
    'zero': {
        'name': 'Fill with Zero',
        'description': 'Replaces all missing values with 0.',
        'when_to_use': 'When missing means "no response" or "zero activity".',
        'pros': 'Simple, explicit meaning.',
        'cons': 'Can distort statistics, may not make sense for ratings.',
        'example': 'Missing values in satisfaction ratings might not mean "0 satisfaction"'
    },
    'constant': {
        'name': 'Fill with Constant',
        'description': 'Replaces missing values with a specific constant value (e.g., 3 for neutral).',
        'when_to_use': 'When you want to assign a specific meaning to missing data.',
        'pros': 'Explicit control, can represent "neutral" or "no opinion".',
        'cons': 'Arbitrary choice can bias results.',
        'example': 'Fill missing satisfaction with 3 (neutral on 1-5 scale)'
    },
    'knn': {
        'name': 'KNN Imputation',
        'description': 'Uses K-Nearest Neighbors to estimate missing values based on similar responses.',
        'when_to_use': 'When variables are related and you have enough complete cases.',
        'pros': 'Considers relationships, more sophisticated.',
        'cons': 'Computationally expensive, requires complete cases.',
        'example': 'If respondents with similar Q1-Q3 scores tend to give similar Q4 scores, use those patterns'
    },
    'group_mean': {
        'name': 'Group Mean Imputation',
        'description': 'Fills missing values with the mean of a specific group (e.g., by organization).',
        'when_to_use': 'When groups have different response patterns.',
        'pros': 'Preserves group differences, more contextual.',
        'cons': 'Requires meaningful grouping variable.',
        'example': 'Use average score from same organization instead of overall average'
    }
}

# ============================================================================
# EDUCATIONAL CONTENT - STATISTICAL METHODS
# ============================================================================

STATISTICAL_METHODS = {
    'descriptive': {
        'name': 'Descriptive Statistics',
        'description': 'Basic summary statistics including mean, median, standard deviation, min, max.',
        'purpose': 'Understand the central tendency and spread of your data.',
        'what_you_learn': 'Average satisfaction levels, consistency of responses, range of ratings.',
        'interpretation': 'High mean = good satisfaction. Low std dev = consistent responses.',
    },
    'normality': {
        'name': 'Normality Test (Shapiro-Wilk)',
        'description': 'Tests if your data follows a normal (bell curve) distribution.',
        'purpose': 'Determine if you can use parametric statistical tests.',
        'what_you_learn': 'Whether scores are normally distributed (most scores near middle vs scattered).',
        'interpretation': 'p > 0.05: Data is normal. p < 0.05: Data is not normal.',
    },
    'correlation': {
        'name': 'Correlation Analysis',
        'description': 'Measures relationships between different questions.',
        'purpose': 'Understand which aspects of the workshop are related.',
        'what_you_learn': 'Do people who rate content high also rate instruction high?',
        'interpretation': 'Values close to 1: Strong positive relationship. Close to -1: Inverse relationship. Close to 0: No relationship.',
    },
    'ttest': {
        'name': 'T-Test',
        'description': 'Compares means between two groups to see if they are significantly different.',
        'purpose': 'Test if satisfaction differs between groups (e.g., organizations).',
        'what_you_learn': 'Is the difference in satisfaction real or just by chance?',
        'interpretation': 'p < 0.05: Groups are significantly different.',
    },
    'anova': {
        'name': 'ANOVA (Analysis of Variance)',
        'description': 'Compares means across multiple groups (3 or more).',
        'purpose': 'Test if satisfaction differs across multiple organizations/groups.',
        'what_you_learn': 'Which groups have significantly different satisfaction levels?',
        'interpretation': 'p < 0.05: At least one group is significantly different.',
    },
    'chi_square': {
        'name': 'Chi-Square Test',
        'description': 'Tests relationships between categorical variables.',
        'purpose': 'See if satisfaction categories relate to organization or other factors.',
        'what_you_learn': 'Are certain organizations more likely to give high ratings?',
        'interpretation': 'p < 0.05: Significant relationship exists.',
    }
}

# ============================================================================
# EDUCATIONAL CONTENT - MACHINE LEARNING MODELS
# ============================================================================

ML_MODELS = {
    'logistic': {
        'name': 'Logistic Regression',
        'description': 'Predicts binary outcomes (satisfied vs unsatisfied) using a linear approach.',
        'purpose': 'Understand which factors predict satisfaction and their importance.',
        'when_to_use': 'Best for binary classification with interpretable results.',
        'strengths': 'Fast, interpretable, shows feature importance, works well with small datasets.',
        'limitations': 'Assumes linear relationships, limited to binary/simple outcomes.',
        'what_you_learn': 'Which questions are most predictive of overall satisfaction?'
    },
    'decision_tree': {
        'name': 'Decision Tree',
        'description': 'Creates a tree of decisions to classify satisfaction levels.',
        'purpose': 'Visual understanding of decision rules for satisfaction.',
        'when_to_use': 'When you want easily interpretable rules.',
        'strengths': 'Easy to visualize and explain, handles non-linear patterns.',
        'limitations': 'Can overfit, unstable with small changes in data.',
        'what_you_learn': 'Clear if-then rules: "If Q1 < 3 and Q2 < 4, then unsatisfied"'
    },
    'random_forest': {
        'name': 'Random Forest',
        'description': 'Combines multiple decision trees for more robust predictions.',
        'purpose': 'Get more accurate predictions by averaging many decision trees.',
        'when_to_use': 'When you want high accuracy and feature importance.',
        'strengths': 'Very accurate, robust, provides feature importance, handles outliers well.',
        'limitations': 'Less interpretable than single tree, slower to train.',
        'what_you_learn': 'Most important factors for satisfaction with high accuracy.'
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'description': 'Builds trees sequentially, each correcting errors of previous ones.',
        'purpose': 'Achieve highest possible prediction accuracy.',
        'when_to_use': 'When accuracy is most important and you have enough data.',
        'strengths': 'Often best performance, captures complex patterns.',
        'limitations': 'Can overfit, requires careful tuning, slower training.',
        'what_you_learn': 'Complex patterns in satisfaction with very high accuracy.'
    },
    'svm': {
        'name': 'Support Vector Machine (SVM)',
        'description': 'Finds the best boundary between satisfied and unsatisfied responses.',
        'purpose': 'Classify with maximum separation between classes.',
        'when_to_use': 'When classes are well-separated and you have clean data.',
        'strengths': 'Effective in high dimensions, memory efficient.',
        'limitations': 'Slow on large datasets, less interpretable.',
        'what_you_learn': 'Clear separation boundary between satisfied and unsatisfied.'
    },
    'knn': {
        'name': 'K-Nearest Neighbors (KNN)',
        'description': 'Predicts satisfaction based on most similar responses.',
        'purpose': 'Use similarity to past responses for prediction.',
        'when_to_use': 'When similar response patterns should give similar outcomes.',
        'strengths': 'Simple concept, no training time, works with irregular patterns.',
        'limitations': 'Slow prediction, sensitive to scale, needs good K value.',
        'what_you_learn': '"People with similar responses tend to have similar satisfaction"'
    },
    'naive_bayes': {
        'name': 'Naive Bayes',
        'description': 'Uses probability theory to predict satisfaction likelihood.',
        'purpose': 'Fast probabilistic classification.',
        'when_to_use': 'When you want quick results and probability estimates.',
        'strengths': 'Very fast, works well with small data, provides probabilities.',
        'limitations': 'Assumes independence (naive assumption), less accurate.',
        'what_you_learn': 'Probability of satisfaction given certain responses.'
    }
}

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Create responses table
    c.execute('''CREATE TABLE IF NOT EXISTS responses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  name TEXT,
                  organization TEXT,
                  q1_label TEXT, q1_score REAL, q1_confidence REAL,
                  q2_label TEXT, q2_score REAL, q2_confidence REAL,
                  q3_label TEXT, q3_score REAL, q3_confidence REAL,
                  q4_label TEXT, q4_score REAL, q4_confidence REAL,
                  q5_label TEXT, q5_score REAL, q5_confidence REAL,
                  overall_score REAL)''')
    
    # Create settings table
    c.execute('''CREATE TABLE IF NOT EXISTS settings
                 (key TEXT PRIMARY KEY,
                  value TEXT)''')
    
    # Create interpretations table
    c.execute('''CREATE TABLE IF NOT EXISTS interpretations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  analysis_type TEXT,
                  interpretation TEXT,
                  timestamp TEXT)''')
    
    conn.commit()
    conn.close()

def save_response(name, org, responses):
    """Save survey response to database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Calculate overall score
    scores = [r['score'] for r in responses if r['score'] is not None]
    overall_score = sum(scores) / len(scores) if scores else None
    
    # Prepare data - 19 values total
    data = [
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        name,
        org,
    ]
    
    for r in responses:
        data.extend([r['label'], r['score'], r['confidence']])
    
    data.append(overall_score)
    
    c.execute('''INSERT INTO responses (timestamp, name, organization, 
                  q1_label, q1_score, q1_confidence,
                  q2_label, q2_score, q2_confidence,
                  q3_label, q3_score, q3_confidence,
                  q4_label, q4_score, q4_confidence,
                  q5_label, q5_score, q5_confidence,
                  overall_score) 
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', data)
    conn.commit()
    conn.close()

def get_all_responses():
    """Get all responses from database"""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM responses", conn)
    conn.close()
    return df

def get_setting(key, default=None):
    """Get setting from database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else default

def save_setting(key, value):
    """Save setting to database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def save_interpretation(analysis_type, interpretation):
    """Save interpretation note"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO interpretations (analysis_type, interpretation, timestamp) VALUES (?, ?, ?)",
              (analysis_type, interpretation, timestamp))
    conn.commit()
    conn.close()

def get_interpretation(analysis_type):
    """Get latest interpretation for analysis type"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT interpretation FROM interpretations WHERE analysis_type=? ORDER BY timestamp DESC LIMIT 1",
              (analysis_type,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else ""

def delete_response(response_id):
    """Delete a response"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM responses WHERE id=?", (response_id,))
    conn.commit()
    conn.close()

def clear_all_responses():
    """Clear all responses"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM responses")
    conn.commit()
    conn.close()

# ============================================================================
# ENHANCED CLEANING FUNCTIONS
# ============================================================================

def apply_imputation(df, strategy='median', constant_value=3, group_col=None):
    """Apply selected imputation strategy"""
    df_clean = df.copy()
    score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
    
    # Ensure numeric
    for col in score_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    if strategy == 'mean':
        for col in score_cols:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    elif strategy == 'median':
        for col in score_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    elif strategy == 'mode':
        for col in score_cols:
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col].fillna(mode_val[0], inplace=True)
    
    elif strategy == 'forward_fill':
        df_clean[score_cols] = df_clean[score_cols].fillna(method='ffill')
    
    elif strategy == 'backward_fill':
        df_clean[score_cols] = df_clean[score_cols].fillna(method='bfill')
    
    elif strategy == 'interpolate':
        for col in score_cols:
            df_clean[col] = df_clean[col].interpolate(method='linear')
    
    elif strategy == 'zero':
        df_clean[score_cols] = df_clean[score_cols].fillna(0)
    
    elif strategy == 'constant':
        df_clean[score_cols] = df_clean[score_cols].fillna(constant_value)
    
    elif strategy == 'knn':
        if ML_AVAILABLE:
            imputer = KNNImputer(n_neighbors=3)
            df_clean[score_cols] = imputer.fit_transform(df_clean[score_cols])
    
    elif strategy == 'group_mean' and group_col and group_col in df_clean.columns:
        for col in score_cols:
            df_clean[col] = df_clean.groupby(group_col)[col].transform(
                lambda x: x.fillna(x.mean())
            )
    
    # Fill any remaining NaN with median as fallback
    for col in score_cols:
        if df_clean[col].isnull().any():
            median = df_clean[col].median()
            df_clean[col].fillna(median if pd.notna(median) else 3.0, inplace=True)
    
    return df_clean

# ============================================================================
# ENHANCED ANALYSIS FUNCTIONS
# ============================================================================

def perform_statistical_analysis(df, methods=['descriptive']):
    """Perform selected statistical analyses"""
    score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
    results = {}
    
    if 'descriptive' in methods:
        results['descriptive'] = {
            'mean': df[score_cols].mean().mean(),
            'std': df[score_cols].std().mean(),
            'median': df[score_cols].median().median(),
            'per_question': {}
        }
        for i, col in enumerate(score_cols, 1):
            results['descriptive']['per_question'][f'Q{i}'] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'median': df[col].median(),
                'min': df[col].min(),
                'max': df[col].max()
            }
    
    if 'normality' in methods and len(df) >= 3:
        results['normality'] = {}
        for i, col in enumerate(score_cols, 1):
            if df[col].std() > 0:
                try:
                    stat, p = stats.shapiro(df[col])
                    results['normality'][f'Q{i}'] = {
                        'statistic': stat,
                        'p_value': p,
                        'is_normal': p > 0.05,
                        'interpretation': f"Data is {'normal' if p > 0.05 else 'not normal'} (p={p:.4f})"
                    }
                except:
                    results['normality'][f'Q{i}'] = {'error': 'Cannot compute'}
    
    if 'correlation' in methods and len(df) >= 3:
        results['correlation'] = df[score_cols].corr()
    
    return results

def train_ml_model(df, model_type='logistic'):
    """Train selected ML model"""
    score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
    
    X = df[score_cols].values
    y = (df['overall_score'] >= 4).astype(int).values
    
    if len(np.unique(y)) < 2:
        return {'error': 'Need both satisfied and unsatisfied responses'}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Select model
    models = {
        'logistic': LogisticRegression(random_state=42, max_iter=1000),
        'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=4),
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'svm': SVC(random_state=42, probability=True),
        'knn': KNeighborsClassifier(n_neighbors=3),
        'naive_bayes': GaussianNB()
    }
    
    model = models.get(model_type, LogisticRegression(random_state=42))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results = {
        'model_type': model_type,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Add feature importance if available
    if hasattr(model, 'feature_importances_'):
        results['feature_importance'] = dict(zip(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], 
                                                  model.feature_importances_.tolist()))
    elif hasattr(model, 'coef_'):
        results['feature_importance'] = dict(zip(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], 
                                                  model.coef_[0].tolist()))
    
    return results

def plot_basic_stats(df):
    """Create basic statistics plots"""
    score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Survey Statistics Dashboard', fontsize=14, fontweight='bold')
    
    # 1. Average scores
    means = [df[col].mean() for col in score_cols]
    colors = ['#2ecc71' if m>=4 else '#f39c12' if m>=3 else '#e74c3c' for m in means]
    axes[0,0].bar(range(5), means, color=colors, edgecolor='black', alpha=0.7)
    axes[0,0].axhline(4, color='green', linestyle='--', alpha=0.5)
    axes[0,0].set_title('Average Scores per Question')
    axes[0,0].set_xticks(range(5))
    axes[0,0].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    axes[0,0].set_ylim(0, 5.5)
    
    # 2. Distribution
    all_scores = df[score_cols].values.flatten()
    axes[0,1].hist(all_scores, bins=5, edgecolor='black', color='#3498db', alpha=0.7)
    axes[0,1].set_title('Score Distribution')
    axes[0,1].set_xlabel('Score')
    
    # 3. Box plot
    df[score_cols].boxplot(ax=axes[1,0])
    axes[1,0].set_title('Score Spread by Question')
    
    # 4. Satisfaction rate
    satisfied = (df['overall_score'] >= 4).sum()
    not_satisfied = (df['overall_score'] < 4).sum()
    axes[1,1].bar(['Not Satisfied', 'Satisfied'], [not_satisfied, satisfied],
                  color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Satisfaction Distribution')
    axes[1,1].set_ylabel('Count')
    
    plt.tight_layout()
    return fig

# ============================================================================
# SIMPLE PREDICTION (Replace with Teachable Machine)
# ============================================================================

def simple_predict(image):
    """Simple prediction - replace with Teachable Machine API call"""
    import random
    gestures = list(GESTURE_MAP.keys())
    gesture = random.choice(gestures)
    confidence = random.uniform(0.7, 0.99)
    return gesture, confidence

# ============================================================================
# ADMIN PANEL
# ============================================================================

def show_method_info(method_dict, method_key):
    """Display educational information about a method"""
    info = method_dict[method_key]
    with st.expander(f"📚 Learn about {info['name']}", expanded=False):
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**{list(info.keys())[3]}:** {list(info.values())[3]}")
        st.markdown(f"**{list(info.keys())[4]}:** {list(info.values())[4]}")
        if len(info) > 5:
            st.markdown(f"**{list(info.keys())[5]}:** {list(info.values())[5]}")
        if 'example' in info:
            st.info(f"💡 Example: {info['example']}")

def admin_panel():
    """Enhanced admin panel with educational features"""
    st.title("🔧 Admin Panel")
    
    # Check admin authentication - NO DEFAULT PASSWORD MESSAGE SHOWN
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.subheader("🔐 Admin Login")
        password = st.text_input("Enter Admin Password:", type="password", key="admin_pw")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Login", type="primary"):
                stored_password = get_setting('admin_password', DEFAULT_ADMIN_PASSWORD)
                if password == stored_password:
                    st.session_state.admin_authenticated = True
                    st.success("✓ Authenticated!")
                    st.rerun()
                else:
                    st.error("✗ Incorrect password")
        
        # NO default password hint shown to users
        return
    
    # Logout button
    if st.button("🚪 Logout"):
        st.session_state.admin_authenticated = False
        st.rerun()
    
    # Tabs for different admin functions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚙️ Settings", 
        "📊 View Data", 
        "🧹 Clean Data", 
        "📈 Statistics", 
        "🤖 Machine Learning"
    ])
    
    # ========== TAB 1: SETTINGS ==========
    with tab1:
        st.subheader("Application Settings")
        
        # Teachable Machine Model URL
        st.markdown("### 🎯 Teachable Machine Model")
        current_model_url = get_setting('model_url', '')
        model_url = st.text_input(
            "Teachable Machine Shareable Link:",
            value=current_model_url,
            help="Paste your Teachable Machine model share link here"
        )
        
        if st.button("💾 Save Model URL"):
            save_setting('model_url', model_url)
            st.success("✓ Model URL saved!")
        
        # Admin Password
        st.markdown("### 🔑 Change Admin Password")
        new_password = st.text_input("New Password:", type="password", key="new_pw")
        confirm_password = st.text_input("Confirm Password:", type="password", key="confirm_pw")
        
        if st.button("🔄 Update Password"):
            if new_password and new_password == confirm_password:
                save_setting('admin_password', new_password)
                st.success("✓ Password updated successfully!")
                st.info("⚠️ Remember your new password - there is no recovery option!")
            else:
                st.error("✗ Passwords don't match or are empty")
        
        # Survey Settings
        st.markdown("### 📋 Survey Settings")
        survey_title = st.text_input("Survey Title:", value=get_setting('survey_title', 'Touchless Satisfaction Survey'))
        if st.button("💾 Save Title"):
            save_setting('survey_title', survey_title)
            st.success("✓ Title saved!")
    
    # ========== TAB 2: VIEW DATA ==========
    with tab2:
        st.subheader("📊 Survey Responses")
        
        df = get_all_responses()
        
        if len(df) == 0:
            st.info("No responses yet. Start collecting survey data!")
        else:
            st.success(f"Total Responses: {len(df)}")
            
            # Display data
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "survey_responses.csv",
                "text/csv",
                key='download-csv'
            )
            
            # Delete options
            st.markdown("### 🗑️ Data Management")
            col1, col2 = st.columns(2)
            
            with col1:
                response_id = st.number_input("Delete Response ID:", min_value=1, step=1)
                if st.button("🗑️ Delete Response"):
                    delete_response(response_id)
                    st.success(f"✓ Deleted response {response_id}")
                    st.rerun()
            
            with col2:
                if st.button("⚠️ Clear All Data", type="secondary"):
                    if st.checkbox("Confirm deletion"):
                        clear_all_responses()
                        st.success("✓ All data cleared")
                        st.rerun()
    
    # ========== TAB 3: CLEAN DATA (ENHANCED) ==========
    with tab3:
        st.subheader("🧹 Data Cleaning with Multiple Strategies")
        
        df = get_all_responses()
        
        if len(df) == 0:
            st.info("No data to clean yet.")
        else:
            st.markdown("### Data Quality Check")
            
            score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
            
            # Show missing values
            missing_counts = {}
            for col in score_cols:
                missing = df[col].isnull().sum()
                missing_counts[col] = missing
            
            if sum(missing_counts.values()) > 0:
                st.warning(f"Found {sum(missing_counts.values())} missing values")
                for col, count in missing_counts.items():
                    if count > 0:
                        st.write(f"  - {col}: {count} missing")
            else:
                st.success("✓ No missing values!")
            
            st.markdown("### 🎓 Select Imputation Strategy")
            
            strategy = st.selectbox(
                "Choose a strategy:",
                options=list(IMPUTATION_STRATEGIES.keys()),
                format_func=lambda x: IMPUTATION_STRATEGIES[x]['name']
            )
            
            # Show educational info
            show_method_info(IMPUTATION_STRATEGIES, strategy)
            
            # Additional parameters for some strategies
            constant_value = 3
            group_col = None
            
            if strategy == 'constant':
                constant_value = st.number_input("Constant value:", min_value=1, max_value=5, value=3)
            
            if strategy == 'group_mean':
                if 'organization' in df.columns:
                    group_col = 'organization'
                    st.info(f"Will use group means by: {group_col}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Apply Imputation", type="primary"):
                    df_clean = apply_imputation(df, strategy, constant_value, group_col)
                    st.success(f"✓ Applied {IMPUTATION_STRATEGIES[strategy]['name']}!")
                    st.dataframe(df_clean[score_cols].describe())
            
            with col2:
                if st.button("💾 Export Cleaned Data"):
                    df_clean = apply_imputation(df, strategy, constant_value, group_col)
                    csv = df_clean.to_csv(index=False)
                    st.download_button(
                        "Download Cleaned CSV",
                        csv,
                        f"cleaned_data_{strategy}.csv",
                        "text/csv",
                        key='download-cleaned'
                    )
    
    # ========== TAB 4: STATISTICS (ENHANCED) ==========
    with tab4:
        st.subheader("📈 Statistical Analysis")
        
        df = get_all_responses()
        
        if len(df) < 2:
            st.info("Need at least 2 responses for statistical analysis")
        else:
            st.markdown("### 🎓 Select Statistical Methods")
            
            methods = st.multiselect(
                "Choose methods to apply:",
                options=list(STATISTICAL_METHODS.keys()),
                default=['descriptive', 'correlation'],
                format_func=lambda x: STATISTICAL_METHODS[x]['name']
            )
            
            # Show info for each selected method
            for method in methods:
                show_method_info(STATISTICAL_METHODS, method)
            
            if st.button("🔬 Run Statistical Analysis", type="primary"):
                with st.spinner("Analyzing..."):
                    # Clean data first
                    df_clean = apply_imputation(df, 'median')
                    results = perform_statistical_analysis(df_clean, methods)
                    
                    # Display results
                    if 'descriptive' in methods and 'descriptive' in results:
                        st.markdown("### 📊 Descriptive Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Overall Mean", f"{results['descriptive']['mean']:.2f}/5.0")
                        with col2:
                            st.metric("Standard Deviation", f"{results['descriptive']['std']:.2f}")
                        with col3:
                            st.metric("Median", f"{results['descriptive']['median']:.2f}")
                        
                        st.markdown("#### Per-Question Statistics")
                        stats_df = pd.DataFrame(results['descriptive']['per_question']).T
                        st.dataframe(stats_df)
                    
                    if 'normality' in methods and 'normality' in results:
                        st.markdown("### 🔔 Normality Tests")
                        for q, result in results['normality'].items():
                            if 'interpretation' in result:
                                st.write(f"**{q}**: {result['interpretation']}")
                    
                    if 'correlation' in methods and 'correlation' in results:
                        st.markdown("### 🔗 Correlation Matrix")
                        st.dataframe(results['correlation'].style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))
                    
                    # Visualizations
                    st.markdown("### 📊 Visualizations")
                    fig = plot_basic_stats(df_clean)
                    st.pyplot(fig)
                    
                    # Interpretation box
                    st.markdown("### 📝 Your Interpretation")
                    interpretation = st.text_area(
                        "Add your interpretation of the statistical results:",
                        value=get_interpretation('statistics'),
                        height=150,
                        key='stats_interp'
                    )
                    if st.button("💾 Save Interpretation", key='save_stats'):
                        save_interpretation('statistics', interpretation)
                        st.success("✓ Interpretation saved!")
    
    # ========== TAB 5: MACHINE LEARNING (ENHANCED) ==========
    with tab5:
        st.subheader("🤖 Machine Learning Analysis")
        
        if not ML_AVAILABLE:
            st.error("ML libraries not available. Install scikit-learn, scipy, matplotlib, seaborn")
            return
        
        df = get_all_responses()
        
        if len(df) < 5:
            st.info(f"Need at least 5 responses for ML analysis. Current responses: {len(df)}")
        else:
            st.markdown("### 🎓 Select Machine Learning Model")
            
            model_type = st.selectbox(
                "Choose a model:",
                options=list(ML_MODELS.keys()),
                format_func=lambda x: ML_MODELS[x]['name']
            )
            
            # Show educational info
            show_method_info(ML_MODELS, model_type)
            
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner(f"Training {ML_MODELS[model_type]['name']}..."):
                    df_clean = apply_imputation(df, 'median')
                    results = train_ml_model(df_clean, model_type)
                    
                    if 'error' in results:
                        st.error(results['error'])
                    else:
                        # Display results
                        st.markdown("### 📊 Model Performance")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{results['accuracy']:.2%}")
                        with col2:
                            st.metric("Precision", f"{results['precision']:.2%}")
                        with col3:
                            st.metric("Recall", f"{results['recall']:.2%}")
                        with col4:
                            st.metric("F1-Score", f"{results['f1']:.2%}")
                        
                        # Feature importance
                        if 'feature_importance' in results:
                            st.markdown("### 📊 Feature Importance")
                            importance_df = pd.DataFrame(
                                results['feature_importance'].items(),
                                columns=['Question', 'Importance']
                            ).sort_values('Importance', ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.barh(importance_df['Question'], importance_df['Importance'])
                            ax.set_xlabel('Importance')
                            ax.set_title(f'Feature Importance - {ML_MODELS[model_type]["name"]}')
                            st.pyplot(fig)
                            
                            st.dataframe(importance_df)
                        
                        # Confusion Matrix
                        if 'confusion_matrix' in results:
                            st.markdown("### 🎯 Confusion Matrix")
                            cm = np.array(results['confusion_matrix'])
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)
                        
                        # Interpretation box
                        st.markdown("### 📝 Your ML Interpretation")
                        ml_interpretation = st.text_area(
                            "Add your interpretation of the ML results:",
                            value=get_interpretation(f'ml_{model_type}'),
                            height=150,
                            key='ml_interp'
                        )
                        if st.button("💾 Save ML Interpretation"):
                            save_interpretation(f'ml_{model_type}', ml_interpretation)
                            st.success("✓ Interpretation saved!")

# ============================================================================
# SURVEY PAGE
# ============================================================================

def survey_page():
    """Main survey interface"""
    survey_title = get_setting('survey_title', 'Touchless Satisfaction Survey')
    st.title(f"✋ {survey_title}")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **Gesture Guide:**
        
        ❤️ Heart = Very Satisfied (5)
        👍 Thumbs Up = Satisfied (4)  
        👎 Thumbs Down = Unsatisfied (2)
        ☝️ Waving = Very Unsatisfied (1)
        ✊ Fist = No Answer
        """)
        
        st.info("Show clear hand gestures for best results!")
    
    # Initialize session
    if 'started' not in st.session_state:
        st.session_state.started = False
        st.session_state.current_q = 0
        st.session_state.responses = []
        st.session_state.completed = False
    
    # Start screen
    if not st.session_state.started:
        st.markdown("## Welcome!")
        st.markdown("Please provide your information to begin the survey.")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your Name:")
        with col2:
            org = st.text_input("Organization:")
        
        if st.button("🚀 Start Survey", type="primary"):
            st.session_state.name = name or "Anonymous"
            st.session_state.org = org or "N/A"
            st.session_state.started = True
            st.rerun()
        return
    
    # Completed screen
    if st.session_state.completed:
        st.success("✅ Survey Complete!")
        st.balloons()
        
        st.markdown("## Your Responses")
        
        df = pd.DataFrame([{
            'Question': f"Q{i+1}",
            'Response': r['label'],
            'Score': r['score'] or 'N/A',
            'Confidence': f"{r['confidence']:.1%}"
        } for i, r in enumerate(st.session_state.responses)])
        
        st.dataframe(df, use_container_width=True)
        
        scores = [r['score'] for r in st.session_state.responses if r['score']]
        if scores:
            avg_score = sum(scores)/len(scores)
            st.metric("Your Average Score", f"{avg_score:.2f}/5.0")
            
            if avg_score >= 4:
                st.success("🎉 Thank you for your positive feedback!")
            elif avg_score >= 3:
                st.info("👍 Thank you for your feedback!")
            else:
                st.warning("We appreciate your honest feedback and will work to improve!")
        
        if st.button("📝 Submit Another Response"):
            st.session_state.started = False
            st.session_state.current_q = 0
            st.session_state.responses = []
            st.session_state.completed = False
            st.rerun()
        return
    
    # Survey in progress
    current_q = st.session_state.current_q
    total_q = len(SURVEY_QUESTIONS)
    
    st.progress(current_q / total_q, text=f"Question {current_q + 1} of {total_q}")
    
    st.markdown(f"## Question {current_q + 1}")
    st.markdown(f"### {SURVEY_QUESTIONS[current_q]}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        img_file = st.camera_input("Show your gesture", key=f"cam_{current_q}")
        
        if img_file:
            image = Image.open(img_file)
            
            with st.spinner("Analyzing gesture..."):
                gesture, confidence = simple_predict(image)
            
            info = GESTURE_MAP[gesture]
            
            st.success(f"Detected: {info['emoji']} {info['label']}")
            st.info(f"Confidence: {confidence:.1%}")
            
            if st.button("✅ Confirm", type="primary"):
                st.session_state.responses.append({
                    'label': info['label'],
                    'score': info['score'],
                    'confidence': confidence
                })
                
                if current_q < total_q - 1:
                    st.session_state.current_q += 1
                    st.rerun()
                else:
                    st.session_state.completed = True
                    
                    # Save to database
                    save_response(
                        st.session_state.name,
                        st.session_state.org,
                        st.session_state.responses
                    )
                    
                    st.rerun()
    
    with col2:
        st.markdown("**Gestures:**")
        for g, info in GESTURE_MAP.items():
            st.write(f"{info['emoji']} {info['label']}")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize database
    init_database()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["📝 Survey", "🔧 Admin Panel"],
            label_visibility="collapsed"
        )
    
    # Route to appropriate page
    if page == "📝 Survey":
        survey_page()
    else:
        admin_panel()

if __name__ == "__main__":
    main()
