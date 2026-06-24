
import traceback
import sys

try:
    exec(open("app_main.py").read())
except Exception as e:
    import streamlit as st
    st.error(f"ERREUR : {e}")
    st.code(traceback.format_exc())
