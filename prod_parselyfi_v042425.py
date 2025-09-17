import streamlit as st
from datetime import datetime

# Set page config with coffee theme
st.set_page_config(
    page_title="Café Corner",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with user profile and ambience controls
with st.sidebar:
    st.title("☕ Café Corner")
    st.markdown("*Where ideas brew and connections pour*")
    
    # User profile section
    st.header("Your Brew Profile")
    if not st.session_state.get("user_logged_in", False):
        with st.form("login_form"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            submit = st.form_submit_button("Steam & Login")
            if submit:
                st.session_state.user_logged_in = True
                st.rerun()
    else:
        st.write(f"Welcome back, {st.session_state.get('username', 'Barista')}!")
        if st.button("Pour New Content", key="pour_new_content"):
            st.session_state.active_tab = "Workshop"
            st.rerun()
        if st.button("Brew Connections", key="brew_connections"):
            st.session_state.active_tab = "Lounge"
            st.rerun()
    
    # Ambience toggle
    st.divider()
    st.subheader("Café Ambience")
    ambient_sound = st.radio(
        "Background sounds:",
        ("Café Chatter", "Lo-Fi Beats", "Quiet")
    )
    volume = st.slider("Volume", 0, 100, 50)

# Main content area
st.title("☕ Welcome to Café Corner")
st.markdown("*Your AI-powered collaborative space*")

# Feature tabs matching the document's four main sections
tab1, tab2, tab3, tab4 = st.tabs([
    "☕ First-Sip Profile", 
    "🪑 Community Counter", 
    "🧪 Brew Lab", 
    "📋 Bulletin Board"
])

# First tab - Profile ("First-Sip" Impression)
with tab1:
    st.header("☕ First-Sip Profile")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Your Profile")
        st.file_uploader("Drag-and-drop a scroll-through of your LinkedIn, Instagram, X, etc.", 
                         type=["mp4", "mov", "jpg", "png", "pdf"],
                         key="profile_upload")
        st.caption("Progress ring will animate like crema rising in a cup.")
        
        if st.button("Extract Profile with AI", key="extract_profile"):
            st.session_state.profile_generated = True
            st.rerun()
    
    with col2:
        st.subheader("Barista Board")
        if st.session_state.get("profile_generated", False):
            with st.container(border=True):
                st.write("Generated Profile Card")
                st.write("Name: Jane Doe")
                st.write("Skills: Data Science, AI, Python")
                st.write("Experience: 5 years")
        
            with st.container(border=True):
                st.write("Edit Your Profile")
                st.text_area("Profile Markdown", height=200, 
                             value="# Jane Doe\n\nSkills: Data Science, AI, Python\n\nExperience: 5 years",
                             key="profile_markdown")
                st.button("Steam & Save", key="save_profile")
        else:
            st.write("Your profile will appear here after extraction.")

# Second tab - Lounge ("Community Counter")
with tab2:
    st.header("🪑 Community Counter")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Chat Tables")
        tables = ["Startup Espresso", "Bioengineering Latte", "AI Cappuccino", "Design Mocha"]
        selected_table = st.selectbox("Join a table", tables, key="table_select")
        
        st.subheader("Find Collaborators")
        st.text_input("Search skills or interests", key="skill_search")
        st.button("Pour-Over Match (15-min call)", key="pour_over_match")
        st.button("Espresso Shot (5-min intro)", key="espresso_shot")
    
    with col2:
        st.subheader(f"Table: {selected_table}")
        
        with st.container(border=True, height=400):
            st.write("Chat messages will appear here.")
        
        col_msg, col_btn = st.columns([4, 1])
        with col_msg:
            st.text_input("Type your message", key="chat_message")
        with col_btn:
            st.button("Send", key="send_message")

# Third tab - Workshop ("Brew Lab")
with tab3:
    st.header("🧪 Brew Lab")
    
    st.subheader("Create a Workshop")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.file_uploader("Upload files for your workshop", 
                         type=["pdf", "jpg", "png", "mp4", "mov"], 
                         accept_multiple_files=True,
                         key="workshop_files")
        st.text_input("Or paste a URL", key="workshop_url")
        st.button("Generate Workshop Outline", key="generate_outline")
    
    with col2:
        st.subheader("Storyboard")
        with st.container(border=True, height=300):
            st.write("Your storyboard will appear here.")
        
        st.write("One-click actions:")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.button("Generate Slides", key="generate_slides")
        with col_b:
            st.button("Export Markdown", key="export_markdown")
        with col_c:
            st.button("Deploy Chatbot", key="deploy_chatbot")

# Fourth tab - Discovery ("Bulletin Board")
with tab4:
    st.header("📋 Bulletin Board")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        st.multiselect("Topics", ["AI", "Data Science", "Design", "Business", "Healthcare"], key="topic_filter")
        st.slider("Recency", 1, 30, 7, help="Days", key="recency_filter")
        st.checkbox("Nearby only", key="nearby_filter")
        st.button("Apply Filters", key="apply_filters")
    
    with col2:
        st.subheader("Workshop Feed")
        
        # Sample workshop cards
        for i in range(3):
            with st.container(border=True):
                col_content, col_actions = st.columns([3, 1])
                with col_content:
                    st.subheader(f"Workshop {i+1}")
                    st.write(f"Sample workshop content description {i+1}")
                    st.write(f"Created by User {i+1}")
                with col_actions:
                    st.button("View", key=f"view_{i}")
                    st.button("Adopt Workshop", key=f"adopt_{i}")

# Footer
st.divider()
st.markdown("☕ **Café Corner** - *Where ideas brew and connections pour*")