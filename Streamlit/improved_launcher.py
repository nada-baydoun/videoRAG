import torch
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torch
import torch
import cv2
import os
import whisper
from tqdm.notebook import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import pandas as pd
from wordcloud import WordCloud
import seaborn as sns
import datetime
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
import pprint
from typing import Dict, List, Any, Tuple, Optional, Union
import time
import faiss
import psycopg2
from psycopg2.extras import execute_values
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import nltk
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')
import sys
import streamlit as st
import base64

# Function to kill any existing Streamlit processes
def kill_streamlit():
    print("Stopping any existing Streamlit processes...")
    try:
        # Find Streamlit processes
        ps_output = subprocess.check_output("ps aux | grep streamlit", shell=True).decode()
        for line in ps_output.split('\n'):
            if 'streamlit run' in line:
                try:
                    pid = int(line.split()[1])
                    print(f"Killing Streamlit process with PID {pid}")
                    os.kill(pid, signal.SIGTERM)
                except:
                    pass
    except:
        print("No Streamlit processes found")

    # Give processes time to terminate
    time.sleep(2)

# Clean up existing files
def clean_up_files():
    print("Cleaning up communication files...")
    files_to_remove = [
        '/content/streamlit_question.txt',
        '/content/streamlit_answer.json'
    ]

    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Removed {file_path}")
            except Exception as e:
                print(f"Could not remove {file_path}: {e}")

# Main launcher function
def launch_system():
    # Clean up
    kill_streamlit()
    clean_up_files()

    # Start the question processor
    print("\n==== Starting Question Processor ====")
    try:
        # Execute the processor in the notebook context
        exec(open('/content/improved_processor.py').read())
        print("✓ Question processor started successfully")
    except Exception as e:
        print(f"❌ Error starting processor: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Start Streamlit
    print("\n==== Starting Streamlit App ====")
    try:
        streamlit_process = subprocess.Popen(
            ["streamlit", "run", "/content/improved_app.py",
             "--server.port=8501",
             "--server.headless=true",
             "--server.enableCORS=false",
             "--server.enableXsrfProtection=false"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Check if Streamlit started successfully
        time.sleep(5)
        if streamlit_process.poll() is not None:
            print(f"❌ Streamlit failed to start (return code: {streamlit_process.returncode})")
            stderr = streamlit_process.stderr.read().decode()
            print(f"Error: {stderr}")
            return False

        print("✓ Streamlit app started successfully")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Open Streamlit in a new window
    print("\n==== Opening Streamlit App ====")
    try:
        output.serve_kernel_port_as_window(8501)
        print("✓ Streamlit app opened in new window")
    except Exception as e:
        print(f"⚠️ Could not open Streamlit in a new window: {e}")
        print("You can access Streamlit at: http://localhost:8501")

    # Success message
    print("\n✅ SYSTEM RUNNING SUCCESSFULLY!")
    print("The question processor is active in the notebook")
    print("The Streamlit app is running and ready to use")
    print("\nYou can now:")
    print("1. Type your own questions in the text input")
    print("2. Select from sample questions in the UI")
    print("3. See results from each retrieval model")
    print("4. Play video segments at the identified timestamps")

    return True

# Run the launcher
if __name__ == "__main__":
    launch_system()
