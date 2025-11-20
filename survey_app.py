# -*- coding: utf-8 -*-
# ============================================================================
# TOUCHLESS SATISFACTION SURVEY - COMPLETE STANDALONE VERSION
# Features:
# - Admin panel with password
# - Teachable Machine model URL configuration
# - Built-in SQLite database (no Google Sheets needed!)
# - Data viewing and management
# - One-click data cleaning
# - Built-in statistical analysis
# - Machine learning with visualization
# - Interpretation notes
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

# Default admin password (change this!)
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
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # timestamp
        name,                                          # name
        org,                                           # organization
    ]
    
    # Add 5 questions × 3 values = 15 values
    for r in responses:
        data.extend([r['label'], r['score'], r['confidence']])
    
    # Add overall score
    data.append(overall_score)
    
    # Total: 3 + 15 + 1 = 19 values
    # INSERT INTO 19 columns (excluding auto-increment id)
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
# ANALYSIS FUNCTIONS
# ============================================================================

def clean_data(df):
    """Clean survey data"""
    df_clean = df.copy()
    
    # Get score columns
    score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
    
    # Replace None with NaN
    for col in score_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Impute missing values with median
    for col in score_cols:
        median = df_clean[col].median()
        if pd.notna(median):
            df_clean[col].fillna(median, inplace=True)
        else:
            df_clean[col].fillna(3.0, inplace=True)
    
    return df_clean

def perform_statistical_analysis(df):
    """Perform statistical tests"""
    score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
    
    results = {}
    
    # Basic statistics
    results['mean'] = df[score_cols].mean().mean()
    results['std'] = df[score_cols].std().mean()
    results['median'] = df[score_cols].median().median()
    
    # Per-question stats
    results['per_question'] = {}
    for i, col in enumerate(score_cols, 1):
        results['per_question'][f'Q{i}'] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'median': df[col].median(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    # Normality tests (if enough data)
    if len(df) >= 3:
        results['normality'] = {}
        for i, col in enumerate(score_cols, 1):
            if df[col].std() > 0:
                try:
                    stat, p = stats.shapiro(df[col])
                    results['normality'][f'Q{i}'] = {'statistic': stat, 'p_value': p, 'normal': p > 0.05}
                except:
                    results['normality'][f'Q{i}'] = {'error': 'Cannot compute'}
    
    # Correlation matrix
    if len(df) >= 3:
        results['correlation'] = df[score_cols].corr()
    
    return results

def perform_ml_analysis(df):
    """Perform machine learning analysis"""
    score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
    
    results = {}
    
    # Prepare data
    X = df[score_cols].values
    y = (df['overall_score'] >= 4).astype(int).values
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results['n_samples'] = len(df)
    results['n_features'] = len(score_cols)
    
    # PCA (if enough samples)
    if len(df) >= 2:
        n_components = min(len(df) - 1, len(score_cols))
        if n_components >= 1:
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            results['pca'] = {
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist()
            }
    
    # Clustering (if enough samples)
    if len(df) >= 3:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        results['clustering'] = {
            'labels': clusters.tolist(),
            'n_clusters': 2
        }
    
    # Classification (if enough samples and both classes)
    if len(df) >= 5 and len(np.unique(y)) > 1:
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            
            # Logistic Regression
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            
            results['classification'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'feature_importance': dict(zip(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], lr.coef_[0].tolist()))
            }
        except:
            pass
    
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
    # TODO: Integrate with Teachable Machine model
    import random
    gestures = list(GESTURE_MAP.keys())
    gesture = random.choice(gestures)
    confidence = random.uniform(0.7, 0.99)
    return gesture, confidence

# ============================================================================
# ADMIN PANEL
# ============================================================================

def admin_panel():
    """Admin panel for configuration and data management"""
    st.title("🔧 Admin Panel")
    
    # Check admin authentication
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.subheader("🔐 Admin Login")
        password = st.text_input("Enter Admin Password:", type="password")
        
        if st.button("Login"):
            stored_password = get_setting('admin_password', DEFAULT_ADMIN_PASSWORD)
            if password == stored_password:
                st.session_state.admin_authenticated = True
                st.success("✓ Authenticated!")
                st.rerun()
            else:
                st.error("✗ Incorrect password")
        
        st.info("Default password: admin123")
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
            st.info(f"Model: {model_url}")
        
        # Admin Password
        st.markdown("### 🔑 Change Admin Password")
        new_password = st.text_input("New Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")
        
        if st.button("🔄 Update Password"):
            if new_password == confirm_password:
                save_setting('admin_password', new_password)
                st.success("✓ Password updated!")
            else:
                st.error("✗ Passwords don't match")
        
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
    
    # ========== TAB 3: CLEAN DATA ==========
    with tab3:
        st.subheader("🧹 Data Cleaning")
        
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
            
            # Cleaning options
            st.markdown("### Cleaning Strategies")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Impute with Median"):
                    df_clean = clean_data(df)
                    st.success("✓ Data cleaned with median imputation")
                    st.dataframe(df_clean[score_cols].describe())
            
            with col2:
                if st.button("📊 Show Statistics"):
                    st.write("Before cleaning:")
                    st.dataframe(df[score_cols].describe())
            
            with col3:
                if st.button("💾 Export Cleaned Data"):
                    df_clean = clean_data(df)
                    csv = df_clean.to_csv(index=False)
                    st.download_button(
                        "Download Cleaned CSV",
                        csv,
                        "cleaned_data.csv",
                        "text/csv"
                    )
    
    # ========== TAB 4: STATISTICS ==========
    with tab4:
        st.subheader("📈 Statistical Analysis")
        
        df = get_all_responses()
        
        if len(df) < 2:
            st.info("Need at least 2 responses for statistical analysis")
        else:
            if st.button("🔬 Run Statistical Analysis"):
                with st.spinner("Analyzing..."):
                    df_clean = clean_data(df)
                    results = perform_statistical_analysis(df_clean)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Mean", f"{results['mean']:.2f}/5.0")
                    with col2:
                        st.metric("Standard Deviation", f"{results['std']:.2f}")
                    with col3:
                        st.metric("Median", f"{results['median']:.2f}")
                    
                    # Per-question stats
                    st.markdown("### Per-Question Statistics")
                    stats_df = pd.DataFrame(results['per_question']).T
                    st.dataframe(stats_df)
                    
                    # Visualizations
                    st.markdown("### Visualizations")
                    fig = plot_basic_stats(df_clean)
                    st.pyplot(fig)
                    
                    # Interpretation box
                    st.markdown("### 📝 Your Interpretation")
                    interpretation = st.text_area(
                        "Add your interpretation of the statistical results:",
                        value=get_interpretation('statistics'),
                        height=150
                    )
                    if st.button("💾 Save Interpretation"):
                        save_interpretation('statistics', interpretation)
                        st.success("✓ Interpretation saved!")
    
    # ========== TAB 5: MACHINE LEARNING ==========
    with tab5:
        st.subheader("🤖 Machine Learning Analysis")
        
        if not ML_AVAILABLE:
            st.error("ML libraries not available. Install scikit-learn, scipy, matplotlib, seaborn")
            return
        
        df = get_all_responses()
        
        if len(df) < 3:
            st.info("Need at least 3 responses for ML analysis")
            st.write(f"Current responses: {len(df)}")
        else:
            if st.button("🚀 Run ML Analysis"):
                with st.spinner("Training models..."):
                    df_clean = clean_data(df)
                    results = perform_ml_analysis(df_clean)
                    
                    # Display results
                    st.markdown("### 📊 Dataset Info")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Samples", results['n_samples'])
                    with col2:
                        st.metric("Features", results['n_features'])
                    
                    # PCA Results
                    if 'pca' in results:
                        st.markdown("### 🔍 Principal Component Analysis")
                        pca_df = pd.DataFrame({
                            'Component': [f'PC{i+1}' for i in range(len(results['pca']['explained_variance']))],
                            'Explained Variance': results['pca']['explained_variance'],
                            'Cumulative Variance': results['pca']['cumulative_variance']
                        })
                        st.dataframe(pca_df)
                    
                    # Classification Results
                    if 'classification' in results:
                        st.markdown("### 🎯 Classification Model Performance")
                        metrics_df = pd.DataFrame({
                            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                            'Value': [
                                results['classification']['accuracy'],
                                results['classification']['precision'],
                                results['classification']['recall'],
                                results['classification']['f1']
                            ]
                        })
                        st.dataframe(metrics_df)
                        
                        st.markdown("### 📊 Feature Importance")
                        importance_df = pd.DataFrame(
                            results['classification']['feature_importance'].items(),
                            columns=['Question', 'Importance']
                        ).sort_values('Importance', ascending=False)
                        st.dataframe(importance_df)
                    
                    # Interpretation box
                    st.markdown("### 📝 Your ML Interpretation")
                    ml_interpretation = st.text_area(
                        "Add your interpretation of the ML results:",
                        value=get_interpretation('machine_learning'),
                        height=150
                    )
                    if st.button("💾 Save ML Interpretation"):
                        save_interpretation('machine_learning', ml_interpretation)
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
            st.metric("Average Score", f"{avg_score:.2f}/5.0")
        
        if st.button("📝 New Response"):
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
            
            with st.spinner("Analyzing..."):
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
