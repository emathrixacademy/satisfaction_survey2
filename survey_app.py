# ============================================================================
# TOUCHLESS SATISFACTION SURVEY - AUTO-SETUP VERSION
# Features:
# - Upload JSON credentials in the app
# - Automatic Google Sheets creation with headers
# - No manual setup needed!
# ============================================================================

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import json
import base64

# Try to import optional dependencies
try:
    import gspread
    from google.oauth2.service_account import Credentials
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Touchless Survey - Auto Setup",
    page_icon="✋",
    layout="wide"
)

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

SHEET_NAME = "Survey_Responses"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def json_to_toml(json_creds):
    """Convert JSON credentials to TOML format string"""
    toml_str = "[google_credentials]\n"
    for key, value in json_creds.items():
        if key == "private_key":
            # Handle multi-line private key
            toml_str += f'{key} = """{value}"""\n'
        else:
            toml_str += f'{key} = "{value}"\n'
    return toml_str

def save_credentials_to_session(json_data):
    """Save credentials to session state"""
    try:
        creds_dict = json.loads(json_data) if isinstance(json_data, str) else json_data
        st.session_state.credentials = creds_dict
        st.session_state.credentials_loaded = True
        return True, "✓ Credentials loaded successfully!"
    except Exception as e:
        return False, f"✗ Error: {str(e)}"

def init_google_sheets():
    """Initialize Google Sheets with credentials from session"""
    if not st.session_state.get('credentials_loaded', False):
        return None, "No credentials loaded"
    
    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds = Credentials.from_service_account_info(
            st.session_state.credentials, 
            scopes=scope
        )
        client = gspread.authorize(creds)
        return client, "Connected"
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_sheet_with_headers(client):
    """Create Google Sheet with proper headers"""
    try:
        # Try to open existing sheet
        try:
            spreadsheet = client.open(SHEET_NAME)
            sheet = spreadsheet.sheet1
            
            # Check if headers exist
            first_row = sheet.row_values(1)
            if not first_row or first_row[0] != 'Timestamp':
                # Add headers
                headers = ['Timestamp', 'Name', 'Org',
                          'Q1', 'S1', 'C1', 'Q2', 'S2', 'C2',
                          'Q3', 'S3', 'C3', 'Q4', 'S4', 'C4',
                          'Q5', 'S5', 'C5']
                sheet.insert_row(headers, 1)
                return sheet, "Headers added to existing sheet"
            return sheet, "Sheet already configured"
            
        except gspread.exceptions.SpreadsheetNotFound:
            # Create new sheet
            spreadsheet = client.create(SHEET_NAME)
            sheet = spreadsheet.sheet1
            
            # Add headers
            headers = ['Timestamp', 'Name', 'Org',
                      'Q1', 'S1', 'C1', 'Q2', 'S2', 'C2',
                      'Q3', 'S3', 'C3', 'Q4', 'S4', 'C4',
                      'Q5', 'S5', 'C5']
            sheet.append_row(headers)
            
            # Share with service account (already has access as creator)
            return sheet, "New sheet created with headers"
            
    except Exception as e:
        return None, f"Error: {str(e)}"

def save_to_sheets(data):
    """Save survey response to Google Sheets"""
    if not SHEETS_AVAILABLE:
        return False, "Google Sheets not available"
    
    if not st.session_state.get('credentials_loaded', False):
        return False, "No credentials loaded"
    
    try:
        client, msg = init_google_sheets()
        if client is None:
            return False, msg
        
        sheet, msg = create_sheet_with_headers(client)
        if sheet is None:
            return False, msg
        
        sheet.append_row(data)
        return True, "✓ Data saved to Google Sheets!"
        
    except Exception as e:
        return False, f"Error saving: {str(e)}"

def simple_predict(image):
    """Simple prediction for demo (replace with actual model)"""
    import random
    gestures = list(GESTURE_MAP.keys())
    gesture = random.choice(gestures)
    confidence = random.uniform(0.7, 0.99)
    return gesture, confidence

# ============================================================================
# SETUP PAGE (NEW!)
# ============================================================================

def setup_page():
    """Setup page for uploading credentials"""
    st.title("🔧 Setup - Google Sheets Configuration")
    
    st.markdown("""
    ### Welcome! Let's set up your survey system.
    
    **What you need:**
    1. Google Cloud service account JSON file
    2. That's it! Everything else is automatic.
    """)
    
    # Check if already set up
    if st.session_state.get('credentials_loaded', False):
        st.success("✓ Credentials already loaded!")
        st.info(f"Service account: {st.session_state.credentials.get('client_email', 'Unknown')}")
        
        if st.button("🔄 Upload Different Credentials"):
            st.session_state.credentials_loaded = False
            st.rerun()
        
        if st.button("▶️ Go to Survey"):
            st.session_state.setup_complete = True
            st.rerun()
        return
    
    # Instructions
    with st.expander("📖 How to get Google Cloud credentials", expanded=False):
        st.markdown("""
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a project (or select existing)
        3. Enable **Google Sheets API** and **Google Drive API**
        4. Go to **IAM & Admin** → **Service Accounts**
        5. Create service account → Add Key → JSON
        6. Download the JSON file
        7. Upload it below!
        """)
    
    # File upload
    st.markdown("### Upload Credentials")
    uploaded_file = st.file_uploader(
        "Upload your service account JSON file",
        type=['json'],
        help="The JSON file you downloaded from Google Cloud"
    )
    
    if uploaded_file is not None:
        try:
            # Read JSON
            json_data = json.load(uploaded_file)
            
            # Show what we got
            st.success("✓ JSON file loaded!")
            st.info(f"Project: {json_data.get('project_id', 'Unknown')}")
            st.info(f"Service account: {json_data.get('client_email', 'Unknown')}")
            
            # Show TOML conversion
            with st.expander("📄 TOML Format (for manual setup)", expanded=False):
                toml_str = json_to_toml(json_data)
                st.code(toml_str, language='toml')
                st.download_button(
                    "Download TOML",
                    toml_str,
                    file_name="streamlit_secrets.toml",
                    mime="text/plain"
                )
            
            # Save to session
            if st.button("✅ Use These Credentials", type="primary"):
                success, msg = save_credentials_to_session(json_data)
                if success:
                    st.success(msg)
                    st.balloons()
                    
                    # Test connection
                    with st.spinner("Testing Google Sheets connection..."):
                        client, msg = init_google_sheets()
                        if client:
                            st.success("✓ Google Sheets connection successful!")
                            
                            # Create sheet with headers
                            sheet, msg = create_sheet_with_headers(client)
                            if sheet:
                                st.success(f"✓ {msg}")
                                st.info(f"Sheet: {SHEET_NAME}")
                                
                                st.session_state.setup_complete = True
                                st.success("🎉 Setup complete! Click below to start surveying.")
                                
                                if st.button("▶️ Start Survey"):
                                    st.rerun()
                            else:
                                st.error(f"Sheet creation issue: {msg}")
                        else:
                            st.error(f"Connection test failed: {msg}")
                else:
                    st.error(msg)
                    
        except Exception as e:
            st.error(f"Error reading JSON file: {str(e)}")
    
    # Manual entry option
    with st.expander("⌨️ Or paste JSON manually", expanded=False):
        json_text = st.text_area(
            "Paste your JSON credentials here",
            height=200,
            placeholder='{"type": "service_account", ...}'
        )
        
        if st.button("Load from Text"):
            try:
                json_data = json.loads(json_text)
                success, msg = save_credentials_to_session(json_data)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# SURVEY PAGE
# ============================================================================

def survey_page():
    """Main survey interface"""
    st.title("✋ Touchless Satisfaction Survey")
    
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
        
        # Setup status
        if st.session_state.get('credentials_loaded', False):
            st.success("✓ Google Sheets connected")
        else:
            st.warning("⚠ Google Sheets not configured")
            if st.button("Go to Setup"):
                st.session_state.setup_complete = False
                st.rerun()
    
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
            
            if avg_score >= 4:
                st.success("🎉 Excellent satisfaction!")
            elif avg_score >= 3:
                st.info("👍 Good satisfaction")
            else:
                st.warning("⚠️ Needs improvement")
        
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
                    
                    # Save to sheets
                    data = [
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        st.session_state.name,
                        st.session_state.org
                    ]
                    for r in st.session_state.responses:
                        data.extend([
                            r['label'],
                            str(r['score']) if r['score'] else 'N/A',
                            f"{r['confidence']:.1%}"
                        ])
                    
                    success, msg = save_to_sheets(data)
                    if success:
                        st.success(msg)
                    else:
                        st.warning(f"Could not save to sheets: {msg}")
                    
                    st.rerun()
    
    with col2:
        st.markdown("**Gestures:**")
        for g, info in GESTURE_MAP.items():
            st.write(f"{info['emoji']} {info['label']}")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if current_q > 0 and st.button("⬅️ Back"):
            st.session_state.current_q -= 1
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.started = False
            st.session_state.current_q = 0
            st.session_state.responses = []
            st.rerun()
    
    with col3:
        if current_q < total_q - 1 and st.button("Skip ➡️"):
            st.session_state.responses.append({
                'label': 'No Answer',
                'score': None,
                'confidence': 1.0
            })
            st.session_state.current_q += 1
            st.rerun()

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize session state
    if 'credentials_loaded' not in st.session_state:
        st.session_state.credentials_loaded = False
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = False
    
    # Route to appropriate page
    if not st.session_state.setup_complete:
        setup_page()
    else:
        survey_page()

if __name__ == "__main__":
    main()
