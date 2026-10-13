# app.py - entry point kept so `streamlit run app.py` keeps working.
# The application lives in deployfinalcode/deploy.py (next to its model files);
# this file used to be an out-of-date copy of it.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "deployfinalcode"))

from deploy import main

if __name__ == "__main__":
    main()
