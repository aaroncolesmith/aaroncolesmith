import streamlit as st


def generate_storyline():
    st.session_state["stage"] = "edit_storyline"


def create_booklet():
    st.session_state["stage"] = "final_parameters"


def final_parameters():
    st.session_state["stage"] = "story_generation"


if "stage" not in st.session_state:
    st.session_state["stage"] = "story_generation"

if st.session_state["stage"] == "story_generation":
    with st.form("story_form"):
        submitted = st.form_submit_button(
            "Generate Storyline", on_click=generate_storyline
        )

elif st.session_state["stage"] == "edit_storyline":
    with st.form("edit_story_form"):
        submitted = st.form_submit_button("Create Booklet", on_click=create_booklet)

elif st.session_state["stage"] == "final_parameters":
    with st.form("final_params_form"):
        submitted = st.form_submit_button("Create Book", on_click=final_parameters)