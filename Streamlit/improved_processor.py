import threading
import threading
import threading
import json
import threading

"""
Improved direct processor for Streamlit questions
"""
import sys
import os
import json
import time
import threading
from video_rag_system import VideoRAGSystem
import threading
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
import sys, importlib

if 'video_rag_system' in sys.modules:
    importlib.reload(sys.modules['video_rag_system'])

# Global variables for tracking processor status
processor_running = True
questions_processed = 0

def run_processor():
    """Process questions from Streamlit using VideoRAGSystem"""
    global processor_running, questions_processed

    print("Starting improved question processor...")

    # Import VideoRAGSystem directly in the notebook's context
    try:
        from video_rag_system import VideoRAGSystem
        print("✅ Successfully imported VideoRAGSystem from notebook")

        def monitor_questions():
            """Thread function to monitor for questions"""
            global processor_running, questions_processed

            print("Starting question monitor thread...")

            while processor_running:
                try:
                    # Check if there's a question file
                    if os.path.exists('/content/streamlit_question.txt'):
                        try:
                            # Read the question
                            with open('/content/streamlit_question.txt', 'r') as f:
                                question = f.read().strip()

                            # Delete the question file immediately
                            os.remove('/content/streamlit_question.txt')

                            if question:
                                print(f"\n==========================================")
                                print(f"Processing question #{questions_processed + 1}: {question}")
                                print(f"==========================================")

                                # Initialize a fresh RAG system for each question
                                rag_system = VideoRAGSystem(
                                    "/content/drive/MyDrive/LLMs and RAG Systems/Assignment_5/Dataset",
                                    "/content/drive/MyDrive/LLMs and RAG Systems/Assignment_5/Dataset/source_video.mp4"
                                )

                                # Process the question
                                print("Using VideoRAGSystem to answer...")
                                start_time = time.time()
                                result = rag_system.answer_question(question)
                                processing_time = time.time() - start_time
                                print(f"Question processed in {processing_time:.2f} seconds!")

                                # Now we need to send the top 3 model answers to the LLM
                                segments = result.get("segments", {})

                                # Prioritize models: FAISS text first, then other FAISS, then others
                                faiss_text_segments = [(name, segment) for name, segment in segments.items()
                                                     if "faiss_text" in name.lower()]
                                other_faiss_segments = [(name, segment) for name, segment in segments.items()
                                                      if "faiss" in name.lower() and "faiss_text" not in name.lower()]
                                other_segments = [(name, segment) for name, segment in segments.items()
                                                 if "faiss" not in name.lower()]

                                # Sort each group by score
                                faiss_text_segments.sort(key=lambda x: x[1].get('score', 0), reverse=True)
                                other_faiss_segments.sort(key=lambda x: x[1].get('score', 0), reverse=True)
                                other_segments.sort(key=lambda x: x[1].get('score', 0), reverse=True)

                                # Combine and take top 3
                                all_segments = faiss_text_segments + other_faiss_segments + other_segments
                                top_segments = all_segments[:3]

                                if top_segments:
                                    print(f"Sending top {len(top_segments)} model answers to LLM...")

                                    # Prepare context for LLM
                                    context = ""
                                    for i, (model_name, segment) in enumerate(top_segments):
                                        text = segment.get('text', 'No text available')
                                        context += f"MODEL {i+1} ({model_name}): {text}\n\n"

                                    # Create LLM prompt
                                    prompt = f"""Based on the following model answers, provide a comprehensive answer to this question: "{question}"

{context}

Answer the question directly based on these model outputs, focusing on the most relevant and accurate information across all models. If the models provide conflicting information, please note that in your answer."""

                                    # Generate answer with LLM
                                    try:
                                        llm_answer = rag_system.generate_with_flan(prompt, max_length=150)
                                        if llm_answer:
                                            print(f"Generated combined answer using all models!")
                                            result["combined_answer"] = llm_answer
                                        else:
                                            print("LLM generation failed, using best model answer as fallback")
                                            # Fallback to best model's text
                                            result["combined_answer"] = top_segments[0][1].get('text', 'No answer available')
                                    except Exception as e:
                                        print(f"Error generating combined answer: {e}")
                                        # Fallback to best model's text
                                        result["combined_answer"] = top_segments[0][1].get('text', 'No answer available')
                                else:
                                    print("No segments available for LLM answer generation")
                                    result["combined_answer"] = "No relevant information found in the video."

                                # Display a summary of the results
                                if result:
                                    answer_preview = result.get('combined_answer', 'No answer available')
                                    if len(answer_preview) > 100:
                                        answer_preview = answer_preview[:100] + "..."
                                    print(f"Combined answer: {answer_preview}")
                                    print(f"Retrieved {len(segments)} segments from different models")
                                else:
                                    print("Warning: VideoRAGSystem returned None or empty result")

                                # Save the result for Streamlit
                                output = {
                                    "answer":           result["combined_answer"],
                                    "segments":         result["segments"],
                                    "system_results":   result["system_results"]
                                }
                                with open('/content/streamlit_answer.json', 'w') as f:
                                    json.dump(result, f)

                                print("Answer saved for Streamlit!")
                                questions_processed += 1
                                print(f"Total questions processed: {questions_processed}")
                                print(f"==========================================\n")

                        except Exception as e:
                            print(f"Error processing question: {e}")
                            import traceback
                            traceback_str = traceback.format_exc()
                            print(f"Traceback: {traceback_str}")

                            # Create error response
                            error_response = {
                                "answer": f"Error processing question: {str(e)}",
                                "has_answer": False,
                                "segments": {},
                                "timespan": None
                            }

                            # Save error response
                            with open('/content/streamlit_answer.json', 'w') as f:
                                json.dump(error_response, f)

                except Exception as e:
                    print(f"Error in monitor loop: {e}")

                # Sleep to avoid high CPU usage
                time.sleep(0.5)

            print("Question monitor thread stopped")

        # Start the monitor thread
        monitor_thread = threading.Thread(target=monitor_questions)
        monitor_thread.daemon = False
        monitor_thread.start()

        print("✅ Question processor is running in a background thread")
        print("Ready to process multiple questions from Streamlit")

        from IPython.display import display, HTML
        display(HTML("""
        <div style="background-color:#d4edda; color:#155724; padding:15px; border-radius:5px; margin-bottom:10px;">
            <h3>✅ Improved Question Handler Active</h3>
            <p>The notebook is now handling questions from the Streamlit app.</p>
            <p>You can now use the Streamlit app to ask multiple questions about the video.</p>
            <p><strong>New:</strong> The system now combines the top 3 model answers using the LLM!</p>
        </div>
        """))

        return monitor_thread

    except Exception as e:
        print(f"❌ Critical error starting processor: {e}")
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")
        return None

# Run the processor
monitor_thread = run_processor()
