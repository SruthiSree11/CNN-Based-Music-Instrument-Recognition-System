"""
Simple Authentication Module for Streamlit App
Handles user login with basic username/password validation
"""

import streamlit as st
import hashlib
import json
from pathlib import Path

# Simple user database (in production, use proper database)
USERS_FILE = Path("users.json")

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def initialize_users():
    """Initialize default users if file doesn't exist"""
    if not USERS_FILE.exists():
        default_users = {
            "admin": {
                "password": hash_password("admin123"),
                "name": "Administrator",
                "role": "admin"
            },
            "demo": {
                "password": hash_password("demo123"),
                "name": "Demo User",
                "role": "user"
            }
        }
        with open(USERS_FILE, 'w') as f:
            json.dump(default_users, f, indent=2)

def verify_user(username: str, password: str) -> dict:
    """Verify user credentials"""
    try:
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
        
        if username in users:
            if users[username]['password'] == hash_password(password):
                return {
                    'success': True,
                    'name': users[username]['name'],
                    'role': users[username]['role']
                }
        return {'success': False, 'message': 'Invalid username or password'}
    except Exception as e:
        return {'success': False, 'message': f'Error: {str(e)}'}

def show_login_page():
    """Display professional login page with cyberpunk theme"""
    
    # Cyberpunk CSS for login page matching main app
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
        
        /* Dark cyberpunk background */
        .main {
            background: linear-gradient(135deg, #0A0E27 0%, #12172E 100%);
        }
        
        /* Hide default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Login container */
        .login-container {
            max-width: 480px;
            margin: 0 auto;
            padding: 48px 40px;
            background: #12172E;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
            margin-top: 60px;
            border: 1px solid rgba(0, 245, 255, 0.2);
            position: relative;
            overflow: hidden;
        }
        
        .login-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #00F5FF 0%, #00B8D4 50%, #0091EA 100%);
            box-shadow: 0 0 20px rgba(0, 245, 255, 0.5);
        }
        
        /* Logo/Icon */
        .login-icon {
            text-align: center;
            font-size: 80px;
            margin-bottom: 24px;
            filter: drop-shadow(0 0 20px rgba(0, 245, 255, 0.5));
        }
        
        /* Login title */
        .login-title {
            text-align: center;
            font-family: 'Orbitron', sans-serif;
            font-size: 36px;
            font-weight: 900;
            background: linear-gradient(135deg, #00F5FF 0%, #00B8D4 50%, #0091EA 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 12px;
            letter-spacing: 2px;
            text-transform: uppercase;
        }
        
        .login-subtitle {
            text-align: center;
            font-size: 15px;
            color: #A0AEC0;
            margin-bottom: 40px;
            font-weight: 500;
            letter-spacing: 1px;
        }
        
        /* Input labels */
        .stTextInput label {
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            margin-bottom: 8px !important;
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            background: #1A1F3A !important;
            border-radius: 10px !important;
            border: 2px solid rgba(0, 245, 255, 0.3) !important;
            padding: 14px !important;
            font-size: 15px !important;
            color: #FFFFFF !important;
            transition: all 0.3s !important;
        }
        
        .stTextInput > div > div > input::placeholder {
            color: #718096 !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #00F5FF !important;
            box-shadow: 0 0 20px rgba(0, 245, 255, 0.3) !important;
            background: #232947 !important;
        }
        
        /* Button styling */
        .stButton > button {
            width: 100% !important;
            background: linear-gradient(135deg, #00F5FF 0%, #00B8D4 50%, #0091EA 100%) !important;
            color: #0A0E27 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 16px !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            cursor: pointer !important;
            transition: all 0.3s !important;
            font-family: 'Orbitron', sans-serif !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            box-shadow: 0 0 20px rgba(0, 245, 255, 0.5) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.02) !important;
            box-shadow: 0 0 30px rgba(0, 245, 255, 0.7), 0 0 60px rgba(0, 245, 255, 0.4) !important;
        }
        
        /* Form container */
        form {
            background: transparent !important;
        }
        
        /* Info box */
        .info-box {
            background: rgba(0, 255, 136, 0.1);
            border-left: 4px solid #00FF88;
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            font-size: 14px;
            color: #A0AEC0;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
        }
        
        .info-box strong {
            color: #00F5FF;
            font-weight: 700;
        }
        
        /* Alert styling */
        .stAlert {
            background: #12172E !important;
            border-radius: 10px !important;
            border-left: 4px solid #00F5FF !important;
            color: #FFFFFF !important;
        }
        
        /* Success/Error messages */
        .stSuccess {
            background: rgba(0, 255, 136, 0.1) !important;
            border-left: 4px solid #00FF88 !important;
            color: #FFFFFF !important;
        }
        
        .stError {
            background: rgba(255, 0, 85, 0.1) !important;
            border-left: 4px solid #FF0055 !important;
            color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Center column for login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Login icon
        st.markdown('<div class="login-icon">🎵</div>', unsafe_allow_html=True)
        
        # Title
        st.markdown('<div class="login-title">Instrument Detector</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">AI-Powered Music Detection System</div>', unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username", key="username_input")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="password_input")
            
            submitted = st.form_submit_button("🔐 Sign In")
            
            if submitted:
                if not username or not password:
                    st.error("⚠️ Please enter both username and password")
                else:
                    with st.spinner("🔄 Verifying credentials..."):
                        result = verify_user(username, password)
                        
                        if result['success']:
                            # Set session state
                            st.session_state['authenticated'] = True
                            st.session_state['username'] = username
                            st.session_state['user_name'] = result['name']
                            st.session_state['user_role'] = result['role']
                            st.success(f"✅ Welcome back, {result['name']}!")
                            st.rerun()
                        else:
                            st.error(f"❌ {result.get('message', 'Login failed')}")
        
        # Demo credentials info
        st.markdown("""
        <div class="info-box">
            <strong>📝 Demo Credentials:</strong><br><br>
            Username: <strong>demo</strong> | Password: <strong>demo123</strong><br>
            Username: <strong>admin</strong> | Password: <strong>admin123</strong>
        </div>
        """, unsafe_allow_html=True)

def check_authentication():
    """Check if user is authenticated"""
    # Initialize users on first run
    initialize_users()
    
    # Check if user is logged in
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    return st.session_state['authenticated']

def logout():
    """Logout user"""
    st.session_state['authenticated'] = False
    if 'username' in st.session_state:
        del st.session_state['username']
    if 'user_name' in st.session_state:
        del st.session_state['user_name']
    if 'user_role' in st.session_state:
        del st.session_state['user_role']
    st.rerun()

def show_user_info():
    """Display user info in sidebar"""
    if 'user_name' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**👤 Logged in as:**")
        st.sidebar.markdown(f"**{st.session_state['user_name']}**")
        st.sidebar.markdown(f"*@{st.session_state['username']}*")
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout()