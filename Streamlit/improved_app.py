import threading
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import base64
from PIL import Image
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
# Configure page
st.set_page_config(page_title="Video Question Answering", layout="wide")

# Setup paths
output_dir = "/content/drive/MyDrive/LLMs and RAG Systems/Assignment_5/Dataset"
video_path = f"{output_dir}/source_video.mp4"
first_set_path = f"{output_dir}/Questions/First_set.json"
second_set_path = f"{output_dir}/Questions/Second_set.json"

# ===== Session State Setup =====
if 'video_timestamp' not in st.session_state:
    st.session_state.video_timestamp = 0
if 'video_key' not in st.session_state:
    st.session_state.video_key = 0
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None
if 'current_segments' not in st.session_state:
    st.session_state.current_segments = {}
if 'current_timespan' not in st.session_state:
    st.session_state.current_timespan = None
if 'answer_history' not in st.session_state:
    st.session_state.answer_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Function to set video timestamp and force refresh
def set_video_position(timestamp):
    st.session_state.video_timestamp = timestamp
    # Increment key to force video reload
    st.session_state.video_key += 1
    st.rerun()

# Function to load questions
def load_questions():
    questions = []

    # Load answerable questions
    try:
        with open(first_set_path, 'r') as f:
            data = json.load(f)
            for q in data.get("qa", []):
                q["answerable"] = True
                questions.append(q)
    except Exception as e:
        st.error(f"Error loading answerable questions: {e}")

    # Load unanswerable questions
    try:
        with open(second_set_path, 'r') as f:
            data = json.load(f)
            for q in data.get("unanswerable", []):
                q["answerable"] = False
                questions.append(q)
    except Exception as e:
        st.error(f"Error loading unanswerable questions: {e}")

    return questions

# Function to process a question
def process_question(question):
    # Set processing state
    st.session_state.processing = True
    
    st.write("➡️ Writing question file to:", '/content/streamlit_question.txt')

    # Write the question to a file for the notebook to pick up
    with open('/content/streamlit_question.txt', 'w') as f:
        f.write(question)

    # Create a placeholder for the processing message
    processing_placeholder = st.empty()
    processing_placeholder.info(f"Processing question: {question}...")

    # Wait for the notebook to process the question
    start_time = time.time()
    max_wait_time = 90  # 90 second timeout

    while time.time() - start_time < max_wait_time:
        if os.path.exists('/content/streamlit_answer.json'):
            try:
                # Read the answer
                with open('/content/streamlit_answer.json', 'r') as f:
                    result = json.load(f)

                # Remove the answer file to avoid reading stale data
                try:
                    os.remove('/content/streamlit_answer.json')
                except:
                    pass

                # Clear the processing message
                processing_placeholder.empty()

                # Update session state
                st.session_state.current_question = question
                st.session_state.current_answer = result.get("answer")
                st.session_state.current_combined_answer = result.get("combined_answer")
                st.session_state.current_segments = result.get("segments", {})
                st.session_state.current_timespan = result.get("timespan")

                # Add to history
                st.session_state.answer_history.append({
                    "question": question,
                    "answer": result.get("answer"),
                    "combined_answer": result.get("combined_answer"),
                    "timespan": result.get("timespan"),
                    "segments": result.get("segments", {})
                })

                # Set video timestamp specifically from FAISS text model if available
                segments = result.get("segments", {})
                faiss_text_segments = [(name, segment) for name, segment in segments.items()
                                      if "faiss_text" in name.lower()]

                if faiss_text_segments:
                    # Sort by score and get the best FAISS text result
                    faiss_text_segments.sort(key=lambda x: x[1].get('score', 0), reverse=True)
                    best_segment = faiss_text_segments[0][1]
                    st.session_state.video_timestamp = best_segment.get('start_time', 0)
                    st.session_state.video_key += 1  # Force video reload
                elif result.get("timespan"):
                    # Fallback to overall timespan if no FAISS text model
                    st.session_state.video_timestamp = result.get("timespan")[0]
                    st.session_state.video_key += 1

                # Clear processing state
                st.session_state.processing = False
                return True
            except Exception as e:
                processing_placeholder.error(f"Error processing answer: {str(e)}")
                time.sleep(0.5)

        time.sleep(0.5)

    # If we reach here, there was a timeout
    processing_placeholder.error("Request timed out. Please check the notebook for errors.")
    st.session_state.processing = False
    return False

# Function to display metrics
def display_metrics(segments):
    if not segments:
        st.warning("No retrieval results available")
        return

    models = []
    scores = []

    for model_name, segment in segments.items():
        models.append(model_name)
        scores.append(segment.get("score", 0))

    # Create dataframe and sort
    df = pd.DataFrame({
        "Model": models,
        "Score": scores
    }).sort_values("Score", ascending=False)

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(df["Model"], df["Score"], color=plt.cm.viridis(np.linspace(0, 0.8, len(models))))

    ax.set_xlabel("Retrieval Model")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Model Performance Comparison")

    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    st.pyplot(fig)

# Function to display segment details
def display_segment(model_name, segment):
    with st.expander(f"{model_name.upper()} - Score: {segment.get('score', 0):.4f}", expanded=True):
        start_time = segment.get('start_time', 0)
        end_time = segment.get('end_time', 0)
        st.markdown(f"**Time Range:** {start_time:.2f}s - {end_time:.2f}s")

        # Play button - simplified to just one button
        if st.button(f"▶️ Play Segment", key=f"play_{model_name}_{start_time}"):
            set_video_position(start_time)

        # Show model's answer more prominently
        st.markdown("**Model's Answer Segment:**")
        text = segment.get('text', 'No text available')
        st.markdown(f"<div style='background-color:#e6f3ff; padding:15px; border-radius:5px; font-size:16px;'>{text}</div>",
                  unsafe_allow_html=True)

        # Display keyframe if available
        if 'keyframe_path' in segment and os.path.exists(segment['keyframe_path']):
            try:
                st.markdown("**Keyframe:**")
                img = Image.open(segment['keyframe_path'])
                st.image(img, caption=f"Keyframe at {segment.get('keyframe_timestamp', segment.get('start_time', 0)):.2f}s", width=400)
            except Exception as e:
                st.error(f"Could not display keyframe: {e}")
        
        # Show the RAG answer 
        rag = segment.get("answer")
        if rag:
            st.markdown("**RAG Answer:**")
            st.markdown(f"> {rag}")


# Function to display timeline visualization
def visualize_timespans(segments):
    if not segments:
        return

    # Extract timestamps
    model_timespans = {}
    min_time = float('inf')
    max_time = 0

    for model_name, segment in segments.items():
        start = segment.get('start_time', 0)
        end = segment.get('end_time', 0)
        model_timespans[model_name] = (start, end)

        if start < min_time:
            min_time = start
        if end > max_time:
            max_time = end

    # Add padding
    padding = (max_time - min_time) * 0.1
    min_time = max(0, min_time - padding)
    max_time = max_time + padding

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 3))

    # Set axis limits
    ax.set_xlim(min_time, max_time)
    ax.set_ylim(0, len(model_timespans) + 1)  # +1 for padding

    # Plot model timespans
    y_pos = 1
    for model_name, (start, end) in model_timespans.items():
        ax.barh(y_pos, end - start, left=start, height=0.5,
                color=plt.cm.viridis(0.2 + 0.7 * y_pos / len(model_timespans)),
                alpha=0.7, label=model_name)
        ax.text(start, y_pos, f"  {model_name}", va='center', ha='left', fontsize=9)
        y_pos += 1

    # Set labels
    ax.set_title('Model Timespan Comparison')
    ax.set_xlabel('Time (seconds)')
    ax.set_yticks([])

    # Remove y-axis
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    st.pyplot(fig)

# Function to calculate combined timespan
def calculate_best_timespan(segments):
    if not segments:
        return None

    weighted_start = 0
    weighted_end = 0
    total_weight = 0
    all_starts = []
    all_ends = []

    # Calculate weighted average
    for model_name, segment in segments.items():
        score = segment.get('score', 0)
        start = segment.get('start_time', 0)
        end = segment.get('end_time', 0)

        all_starts.append(start)
        all_ends.append(end)

        # Use score as weight
        weighted_start += start * score
        weighted_end += end * score
        total_weight += score

    if total_weight > 0:
        # Weighted average based on confidence
        avg_start = weighted_start / total_weight
        avg_end = weighted_end / total_weight

        # Calculate expanded timespan (min start, max end)
        min_start = min(all_starts)
        max_end = max(all_ends)

        return {
            "weighted": (avg_start, avg_end),
            "expanded": (min_start, max_end)
        }
    else:
        return None

# Main UI Layout
st.title("Multimodal RAG System for Video Question Answering")
st.markdown("**Final Assignment Implementation** | LLMs and RAG Systems")

# Two-column layout
col1, col2 = st.columns([3, 2])

with col1:
    # Video player
    st.header("Video Player")
    try:
        # We use the video_key to force reload of the video when timestamp changes
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()
        video_file.close()
        st.video(video_bytes, start_time=int(st.session_state.video_timestamp))

        # Display current timestamp
        if st.session_state.video_timestamp > 0:
            st.info(f"🕒 Video positioned at: {st.session_state.video_timestamp:.2f}s")
    except Exception as e:
        st.error(f"Error loading video: {e}")

    # Display current results
    if st.session_state.current_question:
        st.header("Results")

        # Display current question
        st.subheader(f"Question: {st.session_state.current_question}")

        # Display combined LLM answer if available
        # if hasattr(st.session_state, 'current_combined_answer') and st.session_state.current_combined_answer:
        #     st.subheader("LLM Combined Answer")
        #     st.markdown(f"<div style='background-color:#f0f7ff; padding:15px; border-radius:8px; margin-bottom:20px; font-size:18px;'>{st.session_state.current_combined_answer}</div>",
        #               unsafe_allow_html=True)

        # Calculate best timespan
        best_timespan = calculate_best_timespan(st.session_state.current_segments)

        # Display timespan visualization
        st.subheader("Timeline Visualization")
        if st.session_state.current_segments:
            # Timeline visualization without final answer
            visualize_timespans(st.session_state.current_segments)

        # Display retrieval metrics
        # st.subheader("Retrieval Performance")
        # display_metrics(st.session_state.current_segments)

        # Display individual model results with FAISS first
        st.subheader("Model Results")

        if st.session_state.current_segments:
            # Sort segments by prioritizing FAISS text first, then other FAISS models, then others
            sorted_segments = []

            # First add FAISS text models
            faiss_text_segments = [(name, segment) for name, segment in st.session_state.current_segments.items()
                                 if "faiss_text" in name.lower()]
            faiss_text_segments.sort(key=lambda x: x[1].get('score', 0), reverse=True)
            sorted_segments.extend(faiss_text_segments)

            # Then add other FAISS models
            other_faiss_segments = [(name, segment) for name, segment in st.session_state.current_segments.items()
                                  if "faiss" in name.lower() and "faiss_text" not in name.lower()]
            other_faiss_segments.sort(key=lambda x: x[1].get('score', 0), reverse=True)
            sorted_segments.extend(other_faiss_segments)

            # Then add remaining models
            other_segments = [(name, segment) for name, segment in st.session_state.current_segments.items()
                             if "faiss" not in name.lower()]
            other_segments.sort(key=lambda x: x[1].get('score', 0), reverse=True)
            sorted_segments.extend(other_segments)

            for model_name, segment in sorted_segments:
                display_segment(model_name, segment)
        else:
            st.warning("No model results available")

with col2:
    st.header("Ask a Question")

    # Disable inputs while processing
    if st.session_state.processing:
        input_disabled = True
        input_placeholder = "Processing question... please wait"
    else:
        input_disabled = False
        input_placeholder = "Enter your question about the video"

    # Text input for questions
    question_input = st.text_input("Enter your question about the video",
                                  value="",
                                  key="question_input",
                                  disabled=input_disabled,
                                  placeholder=input_placeholder)

    # Submit button
    submit_button = st.button("Submit Question",
                             key="submit_button",
                             disabled=input_disabled)

    if submit_button and question_input and not st.session_state.processing:
        if process_question(question_input):
            st.rerun()

    # Sample questions from test sets
    st.subheader("Sample Questions")

    # Load test questions
    all_questions = load_questions()

    # Answerable questions
    st.write("**Answerable Questions:**")
    answerable = [q for q in all_questions if q.get("answerable", True)]
    answerable_buttons = []

    for i, q in enumerate(answerable):
        question_text = q.get("question", "")
        sample_button = st.button(f"Q{i+1}: {question_text}",
                                 key=f"ans_btn_{i}",
                                 disabled=input_disabled)

        if sample_button and not st.session_state.processing:
            if process_question(question_text):
                st.rerun()

    # Unanswerable questions
    st.write("**Unanswerable Questions:**")
    unanswerable = [q for q in all_questions if not q.get("answerable", True)]

    for i, q in enumerate(unanswerable):
        question_text = q.get("question", "")
        sample_button = st.button(f"Q{i+1}: {question_text}",
                                 key=f"unans_btn_{i}",
                                 disabled=input_disabled)

        if sample_button and not st.session_state.processing:
            if process_question(question_text):
                st.rerun()

    # Show answer history
    if st.session_state.answer_history:
        st.subheader("Question History")
        for i, item in enumerate(reversed(st.session_state.answer_history)):
            with st.expander(f"Q: {item['question']}"):
                if 'combined_answer' in item and item['combined_answer']:
                    st.markdown(f"**Combined Answer:** {item['combined_answer']}")
                else:
                    st.markdown(f"**Answer:** {item['answer']}")
                if item['timespan']:
                    st.markdown(f"**Timespan:** {item['timespan'][0]:.2f}s - {item['timespan'][1]:.2f}s")
                    if st.button(f"▶️ Play", key=f"history_{i}"):
                        set_video_position(item['timespan'][0])

    # System information
    with st.expander("System Information"):
        # st.markdown("""
        # ### Multimodal RAG System Architecture

        # - **Text Retrieval**: FAISS and pgvector for transcript text  
        # - **Image Retrieval**: CLIP embeddings for visual content  
        # - **Fusion Techniques**: Text and image results combined  
        # - **Lexical Retrieval**: TF-IDF and BM25 for keyword matching  

        # The system uses a text+visual modality approach to give more accurate and relevant answers.
        # """)

        # st.markdown("""
        # ### Evaluation Methodology

        # - **Accuracy**: Correct answers for answerable questions  
        # - **Rejection Quality**: Correctly identifying unanswerable questions  
        # - **Retrieval Effectiveness**: Comparing performance of different retrieval methods  
        # - **Latency**: Processing time for each query  
        # """)

        # --- load and display JSON metrics ---
        # metrics_path = os.path.join(output_dir, "evaluation", "evaluation_metrics.json")
        # try:
        #     with open(metrics_path, 'r') as f:
        #         metrics = json.load(f)
        #     st.subheader("Evaluation Metrics")
        #     st.json(metrics)
        # except Exception as e:
        #     st.error(f"Could not load evaluation metrics: {e}")

        # --- display the two performance charts ---
        st.subheader("Performance Visualizations")
        overall_chart = os.path.join(output_dir, "evaluation", "overall_performance.png")
        latency_chart = os.path.join(output_dir, "evaluation", "retrieval_system_latency.png")

        if os.path.exists(overall_chart):
            st.image(overall_chart, caption="Overall Performance")
        else:
            st.error(f"Missing image: {overall_chart}")

        if os.path.exists(latency_chart):
            st.image(latency_chart, caption="Retrieval System Latency")
        else:
            st.error(f"Missing image: {latency_chart}")

