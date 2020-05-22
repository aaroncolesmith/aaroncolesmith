import pandas as pd
import streamlit as st
from folder.run_function import second
from coronavirus_viz.coronavirus_viz import coronavirus_viz

def main():
    st.write("Hello, this is main")
    second()
    coronavirus_viz()


if __name__ == "__main__":
    #execute
    main()
