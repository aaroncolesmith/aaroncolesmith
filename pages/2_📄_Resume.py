import streamlit as st
import os

def app():
    st.set_page_config(
        page_title="Resume | Aaron Cole Smith",
        page_icon="📄",
        layout="centered"
    )

    # Custom CSS for a premium, clean resume look
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Roboto+Mono:wght@400;500&display=swap');

    .main {
        background-color: #ffffff;
    }
    
    .resume-wrapper {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1a1a1a;
        line-height: 1.6;
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem 0;
    }

    h1 {
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        color: #000000 !important;
        border-bottom: none !important;
    }

    h2 {
        font-weight: 700;
        font-size: 1.25rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        color: #1a1a1a !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 2px solid #eaeaea !important;
        padding-bottom: 0.25rem !important;
    }

    h3 {
        font-weight: 600;
        font-size: 1.1rem !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.5rem !important;
        color: #2d3748 !important;
    }

    p, li {
        font-size: 1.05rem;
        color: #4a5568;
    }

    hr {
        margin: 2rem 0 !important;
        border: none;
        border-top: 1px solid #edf2f7;
    }

    .stMarkdown blockquote {
        background-color: #f7fafc;
        border-left: 4px solid #cbd5e0;
        padding: 0.5rem 1rem;
        margin: 1rem 0;
        font-style: italic;
    }

    code {
        font-family: 'Roboto Mono', monospace !important;
        background-color: #f1f5f9 !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        font-size: 0.9em !important;
        color: #334155 !important;
    }

    /* Print styling */
    @media print {
        .stButton, .stDownloadButton, .stInfo {
            display: none !important;
        }
        .main, .resume-wrapper {
            padding: 0 !important;
            margin: 0 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Read the resume.md file
    resume_path = os.path.join("data", "resume.md")
    
    if os.path.exists(resume_path):
        with open(resume_path, "r") as f:
            resume_text = f.read()
        
        # Wrap everything in a div for styling
        st.markdown('<div class="resume-wrapper">', unsafe_allow_html=True)
        
        # Display the resume content
        st.markdown(resume_text)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button(
                label="📥 Download Markdown",
                data=resume_text,
                file_name="Aaron_Cole_Smith_Resume.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col2:
            st.info("💡 **Tip:** Press `Cmd+P` (Mac) or `Ctrl+P` (Windows) to save this page as a professional PDF.")
            
    else:
        st.error(f"Resume file not found at {resume_path}")

if __name__ == "__main__":
    app()
