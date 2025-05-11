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


class RetrievalSystem:

    def __init__(self, name: str, output_dir: str, modality: str = "text"):
        self.name = name
        self.output_dir = output_dir
        self.modality = modality
        self.text_embeddings_path = f"{output_dir}/embeddings/text/text_embeddings.json"
        self.image_embeddings_path = f"{output_dir}/embeddings/image/temporal_aware_embeddings.json"
        if modality == "text":
            self.embeddings = self._load_text_embeddings()
        elif modality == "image":
            self.embeddings = self._load_image_embeddings()
        elif modality == "fusion":
            self.text_embeddings = self._load_text_embeddings()
            self.image_embeddings = self._load_image_embeddings()
        else:
            raise ValueError(f"Unknown modality: {modality}")

        self.setup_time = 0
        self.is_ready = False

    def _load_text_embeddings(self) -> Dict:
        try:
            with open(self.text_embeddings_path, 'r') as f:
                embeddings = json.load(f)
            print(f"Loaded {len(embeddings)} text embeddings")
            return embeddings
        except FileNotFoundError:
            print(f"Error: Text embeddings not found at {self.text_embeddings_path}")
            return {}

    def _load_image_embeddings(self) -> Dict:
        try:
            with open(self.image_embeddings_path, 'r') as f:
                embeddings = json.load(f)
            print(f"Loaded image embeddings for {len(embeddings)} segments")
            return embeddings
        except FileNotFoundError:
            print(f"Error: Image embeddings not found at {self.image_embeddings_path}")
            return {}

    def build_index(self):
        raise NotImplementedError("Subclasses must implement build_index")

    def search(self, query: Any, top_k: int = 5) -> List[Dict]:
        raise NotImplementedError("Subclasses must implement search")

    def evaluate_latency(self, queries: List[Any], runs: int = 3) -> Dict:
        if not self.is_ready:
            self.build_index()

        total_time = 0
        for _ in range(runs):
            for query in queries:
                start_time = time.time()
                self.search(query, top_k=5)
                total_time += time.time() - start_time

        avg_time = total_time / (runs * len(queries))
        return {
            "system": self.name,
            "modality": self.modality,
            "avg_query_time_ms": avg_time * 1000,
            "setup_time_s": self.setup_time
        }

class FAISSTextRetrieval(RetrievalSystem):

    def __init__(self, output_dir: str):
        super().__init__("FAISS (Flat) - Text", output_dir, modality="text")
        self.index = None
        self.id_mapping = {}

    def build_index(self):
        start_time = time.time()
        embeddings = []
        self.id_mapping = {}
        for idx, (segment_id, data) in enumerate(self.embeddings.items()):
            embedding = np.array(data["embedding"], dtype=np.float32)
            embeddings.append(embedding)
            self.id_mapping[idx] = segment_id
        if embeddings:
            embeddings = np.array(embeddings, dtype=np.float32)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            self.setup_time = time.time() - start_time
            self.is_ready = True
            print(f"FAISS text index built with {len(self.id_mapping)} vectors. Setup time: {self.setup_time:.2f}s")
        else:
            print("No text embeddings to index")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if not self.is_ready:
            self.build_index()
        if self.index is None or len(self.id_mapping) == 0:
            return []
        scores, indices = self.index.search(np.array([query_embedding], dtype=np.float32), min(top_k, len(self.id_mapping)))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.id_mapping):
                segment_id = self.id_mapping[idx]
                if segment_id in self.embeddings:
                    segment_data = self.embeddings[segment_id]
                    results.append({
                        "segment_id": int(segment_id),
                        "score": float(score),
                        "start_time": segment_data["start_time"],
                        "end_time": segment_data["end_time"],
                        "text": segment_data["text"],
                        "modality": "text"
                    })

        return results

class FAISSImageRetrieval(RetrievalSystem):

    def __init__(self, output_dir: str):
        super().__init__("FAISS (Flat) - Image", output_dir, modality="image")
        self.index = None
        self.id_mapping = {}

    def build_index(self):
        start_time = time.time()
        embeddings = []
        self.id_mapping = {}
        idx = 0
        for segment_id, data in self.embeddings.items():
            for keyframe in data["keyframes"]:
                embedding = np.array(keyframe["embedding"], dtype=np.float32)
                embeddings.append(embedding)
                self.id_mapping[idx] = {
                    "segment_id": segment_id,
                    "keyframe_path": keyframe["path"],
                    "timestamp": keyframe["timestamp"]
                }
                idx += 1

        if embeddings:
            embeddings = np.array(embeddings, dtype=np.float32)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            self.setup_time = time.time() - start_time
            self.is_ready = True
            print(f"FAISS image index built with {len(self.id_mapping)} vectors. Setup time: {self.setup_time:.2f}s")
        else:
            print("No image embeddings to index")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if not self.is_ready:
            self.build_index()
        if self.index is None or len(self.id_mapping) == 0:
            return []
        scores, indices = self.index.search(np.array([query_embedding], dtype=np.float32), min(top_k * 2, len(self.id_mapping)))
        segment_scores = {}
        segment_keyframes = {}
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.id_mapping):
                info = self.id_mapping[idx]
                segment_id = info["segment_id"]
                if segment_id not in segment_scores or score > segment_scores[segment_id]:
                    segment_scores[segment_id] = score
                    segment_keyframes[segment_id] = info
        results = []
        for segment_id, score in sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            if segment_id in self.embeddings:
                segment_data = self.embeddings[segment_id]
                keyframe_info = segment_keyframes[segment_id]
                results.append({
                    "segment_id": int(segment_id),
                    "score": float(score),
                    "start_time": segment_data["start_time"],
                    "end_time": segment_data["end_time"],
                    "keyframe_path": keyframe_info["keyframe_path"],
                    "keyframe_timestamp": keyframe_info["timestamp"],
                    "modality": "image"
                })
                if "text" in segment_data:
                    results[-1]["text"] = segment_data["text"]

        return results

class FAISSFusionRetrieval(RetrievalSystem):

    def __init__(self, output_dir: str):
        super().__init__("FAISS (Flat) - Fusion", output_dir, modality="fusion")
        self.text_index = None
        self.image_index = None
        self.text_id_mapping = {}
        self.image_id_mapping = {}
        self.visual_terms = [
            "show", "display", "image", "graph", "diagram", "slide",
            "visual", "screen", "figure", "picture", "drawing", "see", "look"
        ]
        self.technical_terms = [
            "token", "sliding", "jumping", "algorithm", "complexity",
            "graph", "reconfiguration", "parameterized", "theorem", "explain",
            "definition", "concept", "technique"
        ]

    def analyze_query_type(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        if any(term in query_lower for term in self.visual_terms):
            return {"type": "Visual", "text_weight": 0.3, "image_weight": 0.7}
        elif any(term in query_lower for term in self.technical_terms):
            return {"type": "Technical", "text_weight": 0.8, "image_weight": 0.2}
        else:
            return {"type": "General", "text_weight": 0.6, "image_weight": 0.4}

    def build_index(self):
        start_time = time.time()
        text_embeddings = []
        self.text_id_mapping = {}
        for idx, (segment_id, data) in enumerate(self.text_embeddings.items()):
            embedding = np.array(data["embedding"], dtype=np.float32)
            text_embeddings.append(embedding)
            self.text_id_mapping[idx] = segment_id
        if text_embeddings:
            text_embeddings = np.array(text_embeddings, dtype=np.float32)
            self.text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
            self.text_index.add(text_embeddings)
            print(f"FAISS text index built with {len(self.text_id_mapping)} vectors.")
        else:
            self.text_index = None
            print("No text embeddings available to build text index.")
        image_embeddings = []
        self.image_id_mapping = {}
        idx = 0
        for segment_id, data in self.image_embeddings.items():
            for keyframe in data["keyframes"]:
                embedding = np.array(keyframe["embedding"], dtype=np.float32)
                image_embeddings.append(embedding)
                self.image_id_mapping[idx] = {
                    "segment_id": segment_id,
                    "keyframe_path": keyframe["path"],
                    "timestamp": keyframe["timestamp"]
                }
                idx += 1
        if image_embeddings:
            image_embeddings = np.array(image_embeddings, dtype=np.float32)
            self.image_index = faiss.IndexFlatIP(image_embeddings.shape[1])
            self.image_index.add(image_embeddings)
            print(f"FAISS image index built with {len(self.image_id_mapping)} vectors.")
        else:
            self.image_index = None
            print("No image embeddings available to build image index.")
        self.setup_time = time.time() - start_time
        self.is_ready = True
        print(f"FAISS fusion indices built. Setup time: {self.setup_time:.2f}s")

    def normalize_scores(self, score_dict: Dict[int, float]) -> Dict[int, float]:
        if not score_dict:
            return {}
        values = list(score_dict.values())
        max_score = max(values)
        min_score = min(values)
        if max_score == min_score:
            return {k: 1.0 for k in score_dict}
        return {k: (v - min_score) / (max_score - min_score) for k, v in score_dict.items()}

    def search(self, query_data: Dict, top_k: int = 5) -> List[Dict]:
        if not self.is_ready:
            self.build_index()
        query_text = query_data["text"]
        query_text_embedding = query_data["text_embedding"]
        query_image_embedding = query_data["image_embedding"]
        query_analysis = self.analyze_query_type(query_text)
        text_weight = query_analysis["text_weight"]
        image_weight = query_analysis["image_weight"]
        print(f"Query type: {query_analysis['type']}")
        print(f"Weights: Text {text_weight:.2f}, Image {image_weight:.2f}")
        text_scores = {}
        if self.text_index is not None:
            scores, indices = self.text_index.search(
                np.array([query_text_embedding], dtype=np.float32),
                min(top_k * 2, len(self.text_id_mapping))
            )
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.text_id_mapping):
                    segment_id = self.text_id_mapping[idx]
                    text_scores[segment_id] = float(score)
        image_scores = {}
        segment_keyframes = {}
        if self.image_index is not None:
            scores, indices = self.image_index.search(
                np.array([query_image_embedding], dtype=np.float32),
                min(top_k * 3, len(self.image_id_mapping))
            )
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.image_id_mapping):
                    info = self.image_id_mapping[idx]
                    segment_id = info["segment_id"]
                    if segment_id not in image_scores or score > image_scores[segment_id]:
                        image_scores[segment_id] = float(score)
                        segment_keyframes[segment_id] = info
        text_scores = self.normalize_scores(text_scores)
        image_scores = self.normalize_scores(image_scores)
        combined_scores = {}
        for segment_id, score in text_scores.items():
            combined_scores[segment_id] = score * text_weight
        for segment_id, score in image_scores.items():
            if segment_id in combined_scores:
                combined_scores[segment_id] += score * image_weight
            else:
                combined_scores[segment_id] = score * image_weight
        top_segments = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for segment_id, score in top_segments:
            result = {
                "segment_id": int(segment_id),
                "score": float(score),
                "fusion_weights": {"text": text_weight, "image": image_weight},
                "modality": "fusion"
            }
            if segment_id in self.text_embeddings:
                segment_data = self.text_embeddings[segment_id]
                result.update({
                    "start_time": segment_data["start_time"],
                    "end_time": segment_data["end_time"],
                    "text": segment_data["text"],
                    "text_score": text_scores.get(segment_id, 0.0) * text_weight
                })
            if segment_id in segment_keyframes:
                keyframe_info = segment_keyframes[segment_id]
                result.update({
                    "keyframe_path": keyframe_info["keyframe_path"],
                    "keyframe_timestamp": keyframe_info["timestamp"],
                    "image_score": image_scores.get(segment_id, 0.0) * image_weight
                })
            results.append(result)
        return results


class TFIDF:
    def __init__(self, transcript_path):
        self.transcript_path = transcript_path
        self.segments = []
        self.vectorizer = None
        self.document_vectors = None
        self.is_ready = False

    def load_segments(self):
        with open(self.transcript_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            self.segments = data
        elif "segments" in data:
            self.segments = []
            for i, seg in enumerate(data["segments"]):
                self.segments.append({
                    "id": i,
                    "start_time": seg["start"],
                    "end_time": seg["end"],
                    "text": seg["text"]
                })
        print(f"Loaded {len(self.segments)} segments")

    def build_index(self):
        if not self.segments:
            self.load_segments()
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [segment["text"] for segment in self.segments]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            token_pattern=r'\b\w+[\w\'-]*\w+\b|[\w\'-]+'
        )
        self.document_vectors = self.vectorizer.fit_transform(texts)
        self.is_ready = True
        vocab_size = len(self.vectorizer.get_feature_names_out())
        print(f"TF-IDF index built with {len(texts)} documents and {vocab_size} terms")

    def search(self, query, top_k=5):
        if not self.is_ready:
            self.build_index()
        query_vector = self.vectorizer.transform([query])
        scores = (query_vector @ self.document_vectors.T).toarray()[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            segment = self.segments[idx]
            if scores[idx] > 0:
                results.append({
                    "segment_id": segment["id"],
                    "score": float(scores[idx]),
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "text": segment["text"],
                    "modality": "text"
                })
        return results

class BM25(RetrievalSystem):
    def __init__(self, output_dir: str):
        super().__init__("BM25", output_dir, modality="text")
        self.bm25 = None
        self.segment_ids = []
        self.tokenized_corpus = []

    def tokenize_text(self, text):
        import re
        from nltk.corpus import stopwords
        try:
            stops = stopwords.words('english')
        except:
            import nltk
            nltk.download('stopwords')
            stops = stopwords.words('english')
        tokens = re.findall(r'\b\w+[\w\'-]*\w+\b|[\w\'-]+', text.lower())
        tokens = [t for t in tokens if t not in stops and len(t) > 1]
        return tokens

    def build_index(self):
        start_time = time.time()
        texts = []
        self.segment_ids = []
        for segment_id, data in self.embeddings.items():
            texts.append(data["text"])
            self.segment_ids.append(segment_id)
        self.tokenized_corpus = [self.tokenize_text(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.setup_time = time.time() - start_time
        self.is_ready = True
        print(f"Improved BM25 index built with {len(texts)} documents. Setup time: {self.setup_time:.2f}s")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.is_ready:
            self.build_index()
        tokenized_query = self.tokenize_text(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                segment_id = self.segment_ids[idx]
                segment_data = self.embeddings[segment_id]
                results.append({
                    "segment_id": int(segment_id),
                    "score": float(scores[idx]),
                    "start_time": segment_data["start_time"],
                    "end_time": segment_data["end_time"],
                    "text": segment_data["text"],
                    "modality": "text"
                })
        return results



class VideoRAGSystem:
    def __init__(self, output_dir, video_path):
        self.output_dir = output_dir
        self.video_path = video_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading FLAN-T5-large model")
        try:
            self.llm_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
            self.llm_model = T5ForConditionalGeneration.from_pretrained(
                "google/flan-t5-large",
                device_map="auto"
            )
            self.model_size = "large"
            print(f"✅ FLAN-T5-large loaded successfully")
            self.model_loaded = True
        except Exception as e:
            print(f"⚠️ Error loading FLAN-T5-large: {e}")
            print("Trying to load FLAN-T5-base as fallback...")
            try:
                self.llm_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
                self.llm_model = T5ForConditionalGeneration.from_pretrained(
                    "google/flan-t5-base",
                    device_map="auto"
                )
                self.model_size = "base"
                print("✅ FLAN-T5-base loaded successfully as fallback")
                self.model_loaded = True
            except Exception as e2:
                print(f"⚠️ Error loading FLAN-T5-base: {e2}")
                print("Trying to load FLAN-T5-small...")
                try:
                    self.llm_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
                    self.llm_model = T5ForConditionalGeneration.from_pretrained(
                        "google/flan-t5-small",
                        device_map="auto"
                    )
                    self.model_size = "small"
                    print("✅ FLAN-T5-small loaded successfully")
                    self.model_loaded = True
                except Exception as e3:
                    print(f"❌ Error loading any FLAN-T5 model: {e3}")
                    print("System will operate in retrieval-only mode")
                    self.llm_tokenizer = None
                    self.llm_model = None
                    self.model_size = None
                    self.model_loaded = False

        print("Loading embedding models...")
        try:
            self.text_model = SentenceTransformer("ibm-granite/granite-embedding-107m-multilingual")
            print("✅ SentenceTransformer loaded successfully")
        except Exception as e:
            print(f"❌ Error loading SentenceTransformer: {e}")
            print("Trying to load a fallback embedding model...")
            try:
                self.text_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
                print("✅ Fallback SentenceTransformer loaded")
            except Exception as e2:
                print(f"❌ Critical error: Could not load any embedding model: {e2}")
                raise RuntimeError("Cannot continue without an embedding model")

        try:
            self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✅ CLIP model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading CLIP model: {e}")
            print("System will operate without image embeddings")
            self.image_model = None
            self.image_processor = None

        print("Initializing retrieval systems...")
        self.retrieval_systems = {}

        print("Setting up FAISS systems...")
        self.retrieval_systems["faiss_text"] = FAISSTextRetrieval(output_dir)
        if self.image_model is not None:
            self.retrieval_systems["faiss_image"] = FAISSImageRetrieval(output_dir)
            self.retrieval_systems["faiss_fusion"] = FAISSFusionRetrieval(output_dir)

        print("Setting up lexical retrieval systems...")
        try:
            transcript_path = f"{output_dir}/processed/segments_with_keyframes.json"
            self.retrieval_systems["tfidf"] = TFIDF(transcript_path)
            print("✅ TFIDF initialized")

            print("Initializing BM25...")
            self.retrieval_systems["bm25"] = BM25(output_dir)
            print("✅ BM25 initialized")
        except Exception as e:
            print(f"⚠️ Error initializing lexical systems: {e}")

        for name, system in list(self.retrieval_systems.items()):
            try:
                print(f"Building index for {name}...")
                system.build_index()
                print(f"✅ {name} index built successfully")
            except Exception as e:
                print(f"❌ Error building index for {name}: {e}")
                if name in self.retrieval_systems:
                    del self.retrieval_systems[name]

        print(f"✅ Successfully initialized {len(self.retrieval_systems)} retrieval systems")
        self.generation_log = []

    def generate_with_flan(self, prompt, max_length=150):
        if not self.model_loaded:
            return "No language model available"

        try:
            inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
            if hasattr(self.llm_tokenizer, 'model_max_length') and inputs.input_ids.shape[1] > self.llm_tokenizer.model_max_length:
                print(f"⚠️ Prompt exceeds model's token limit, truncating...")
                inputs = self.llm_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.llm_tokenizer.model_max_length - 10
                ).to(self.llm_model.device)
            with torch.no_grad():
                output = self.llm_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    num_beams=3,
                    do_sample=False,
                    temperature=0.3,
                    repetition_penalty=1.5
                )

            response = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)
            return response

        except Exception as e:
            print(f"Error generating with FLAN-T5: {e}")
            return None

    def rewrite_query(self, query):
        prompt = f"""Given this question: "{query}"
        Rewrite it to be clear and concise for a retrieval system searching through video transcripts.
        If the original query is already clear, return EXACTLY the same query.
        Focus on identifying key search terms and maintaining the original intent.
        Rewritten query:"""

        try:
            start_time = time.time()
            rewritten = self.generate_with_flan(prompt, max_length=60)
            elapsed = time.time() - start_time
            # similarity = difflib.SequenceMatcher(None, query.lower(), rewritten.lower()).ratio()
            # if similarity > 0.8:
            #     return query
            print(f"Rewritten from '{query}' to '{rewritten}' in {elapsed:.2f}s")
            return rewritten.strip()
        except Exception as e:
            print(f"Error rewriting query: {e}")
            return query

    def expand_query(self, query):
        return query

    def embed_query(self, query):
        result = {"text": query}
        text_embedding = self.text_model.encode(query)
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
        result["text_embedding"] = text_embedding

        if self.image_model is not None and self.image_processor is not None:
            max_tokens = 70
            if len(query.split()) > 20:
                truncated_query = " ".join(query.split()[:20])
                print(f"Query truncated for CLIP: {truncated_query}")
            else:
                truncated_query = query

            inputs = self.image_processor(text=[truncated_query], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                image_embedding = self.image_model.get_text_features(**inputs)[0].cpu().numpy()
            image_embedding = image_embedding / np.linalg.norm(image_embedding)
            result["image_embedding"] = image_embedding

        return result

    def retrieve_with_system(self, name, system, query, embedded_query, top_k=1):
        try:
            start_time = time.time()
            if name == "faiss_text":
                results = system.search(embedded_query["text_embedding"], top_k=top_k)
            elif name == "faiss_image":
                if "image_embedding" not in embedded_query:
                    return []
                results = system.search(embedded_query["image_embedding"], top_k=top_k)
            elif name == "tfidf":
                results = system.search(query, top_k=top_k)
            elif name == "bm25":
                results = system.search(query, top_k=top_k)
            else:
                results = system.search(embedded_query, top_k=top_k)
            retrieval_time = (time.time() - start_time) * 1000
            if results:
                for result in results:
                    result['retrieval_time'] = retrieval_time
                    if 'start_time' not in result:
                        result['start_time'] = 0.0
                    if 'end_time' not in result:
                        result['end_time'] = 0.0
                    if 'text' not in result:
                        result['text'] = "No text available for this segment"
                    if 'score' not in result:
                        result['score'] = 1.0

            return results
        except Exception as e:
            print(f"Error with {name} retrieval: {e}")
            return []

    def retrieve_all(self, query, top_k=1):
        results = {}
        embedded_query = self.embed_query(query)

        retriever_weights = {
            "tfidf": 2.0,
            "bm25": 2.0,
            "faiss_text": 0.6,
            "faiss_image": 0.5,
            "faiss_fusion": 0.7
        }

        for name, system in self.retrieval_systems.items():
            try:
                print(f"Retrieving with {name}...")
                system_results = self.retrieve_with_system(name, system, query, embedded_query, top_k=1)
                if system_results and len(system_results) > 0:
                    segment = system_results[0]
                    if segment.get('text') == "No text available for this segment" or len(segment.get('text', '').strip()) < 5:
                        print(f"  ✗ Result from {name} has no meaningful text")
                        continue
                    weight = retriever_weights.get(name, 1.0)
                    segment['original_score'] = segment['score']
                    segment['score'] = segment['score'] * weight
                    segment['retriever'] = name
                    results[name] = segment
                    print(f"  ✓ Got result from {name} at time {segment['start_time']:.2f}s-{segment['end_time']:.2f}s")
                    print(f"    Score: {segment['score']:.4f} (original: {segment['original_score']:.4f}, weight: {weight})")
                else:
                    print(f"  ✗ No results from {name}")
            except Exception as e:
                print(f"  ⚠️ Error with {name} retrieval: {e}")
        return results

    def format_segments_for_llm(self, segments_dict, max_context_length=1200):
        if not segments_dict:
            return ""
        segments = [(name, segment) for name, segment in segments_dict.items()]
        segments.sort(key=lambda x: x[1].get('score', 0), reverse=True)
        context = ""
        current_length = 0
        for i, (system_name, segment) in enumerate(segments):
            text = segment.get('text', "No text available for this segment")
            segment_text = f"{text}\n\n"
            segment_length = len(segment_text)
            if current_length + segment_length <= max_context_length:
                context += segment_text
                current_length += segment_length
            else:
                remaining_space = max_context_length - current_length
                if remaining_space > 50:
                    shortened_text = text[:remaining_space - 5] + "..."
                    context += f"{shortened_text}\n\n"
                break
        return context

    def expand_timespan(self, start_time, end_time, buffer_seconds=10.0):
        return (max(0, start_time - buffer_seconds), end_time + buffer_seconds)

    def generate_answer(self, query, segments_dict):
        if not segments_dict:
            return {
                "answer": "I couldn't find relevant information in the video to answer your question.",
                "has_answer": False,
                "segments": {},
                "timespan": None
            }
        context = self.format_segments_for_llm(segments_dict)
        try:
            if self.model_loaded:
              #4. If the CONTEXT doesn't contain ANY information to answer, respond with "i don't have enough information"
                prompt = f"""You are an AI assistant answering questions about a video based ONLY on the provided context from the video transcript.

CONTEXT FROM VIDEO:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer ONLY based on the information in the CONTEXT above. NOTE: The context might include more information than necessary, only use WHAT IS NECESSARY
2. Be concise and specific, focusing only on what's mentioned in the context.
3. Do not use knowledge from outside the provided context.

YOUR ANSWER:"""
                print("\n📝 Answer generation prompt:")
                print("-" * 80)
                print(f"Context length: {len(context)} characters, {len(context.split())} words")
                print(f"Question: {query}")
                print("-" * 80)
                gen_start_time = time.time()
                answer_text = self.generate_with_flan(prompt, max_length=120)
                gen_time = time.time() - gen_start_time
                print(f"✨ Generated answer in {gen_time:.2f}s: {answer_text}")
                print("-" * 80)
                self.generation_log.append({
                    "type": "answer_generation",
                    "query": query,
                    "answer": answer_text,
                    "gen_time": gen_time,
                    "model": f"flan-t5-{self.model_size}",
                    "num_segments": len(segments_dict),
                    "context_length": len(context)
                })
            else:
                first_segment = list(segments_dict.values())[0]
                answer_text = first_segment.get('text', "No text available")
                print("⚠️ Using raw segment text as no model is available")
            answer_text = self.clean_response(answer_text)
            no_answer_phrases = [
                "i don't have enough information",
                "i do not have enough information"
            ]
            has_answer = not any(phrase in answer_text.lower() for phrase in no_answer_phrases)
            if segments_dict:
                first_segment = list(segments_dict.values())[0]
                timespan = (
                    first_segment.get('start_time', 0),
                    first_segment.get('end_time', 0)
                )
                expanded_timespan = self.expand_timespan(*timespan)
            else:
                timespan = None
                expanded_timespan = None

            return {
                "answer": answer_text,
                "has_answer": has_answer,
                "segments": segments_dict,
                "timespan": timespan,
                "expanded_timespan": expanded_timespan
            }

        except Exception as e:
            print(f"Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error generating answer: {e}",
                "has_answer": False,
                "segments": segments_dict,
                "timespan": None
            }

    def clean_response(self, response):
        if not response:
            return "No response generated"
        response = response.strip()
        if "Do not include segment numbers" in response:
            response = response.split("Do not include segment numbers")[0].strip()
        response = re.sub(r'SEGMENT \d+:', '', response)
        response = re.sub(r'SEGMENT \d+ \(\d+\.\d+s - \d+\.\d+s\):', '', response)
        response = re.sub(r'\[Video Segment \d+\]', '', response)
        response = re.sub(r'\[Segment \d+\]', '', response)
        response = re.sub(r'\(Timestamp: \d+\.\d+s - \d+\.\d+s\)', '', response)
        response = re.sub(r'\(\s*\):', '', response)
        response = re.sub(r'\(\s*\)', '', response)
        response = re.sub(r'\s+', ' ', response)
        words = response.split()
        for length in range(5, 20):
            if len(words) < length*2:
                continue
            for i in range(len(words) - length*2 + 1):
                phrase1 = " ".join(words[i:i+length])
                phrase2 = " ".join(words[i+length:i+length*2])
                if phrase1 == phrase2:
                    response = response.replace(phrase1 + " " + phrase2, phrase1)
                    return self.clean_response(response)
        response = re.sub(r'\s+', ' ', response).strip()
        return response

    def answer_question(self, query, top_k=1):
        print(f"\n📝 Processing query: {query}")
        sys.stdout.flush()
        rewritten_query = self.rewrite_query(query)
        if rewritten_query != query:
            print(f"🔄 Rewritten query: {rewritten_query}")
        query = rewritten_query
        embedded_query = self.embed_query(query)
        print("\n🔍 Results per retrieval system:")
        system_results = {}
        all_segments = {}
        for name, system in self.retrieval_systems.items():
            print(f"\n=== {name.upper()} ===")
            try:
                retrieval_results = self.retrieve_with_system(name, system, query, embedded_query, top_k=top_k)
                if not retrieval_results or len(retrieval_results) == 0:
                    print(f"❌ No results found with {name}")
                    system_results[name] = {
                        "answer": f"Could not find relevant information using {name}.",
                        "has_answer": False,
                        "timespan": None,
                        "segments": {}
                    }
                    continue
                top_segment = retrieval_results[0]
                print(f"✓ Found segment at time {top_segment['start_time']:.2f}s-{top_segment['end_time']:.2f}s")
                print(f"  Score: {top_segment['score']:.4f}")
                
                print(f"\n📑 Retrieved text:")
                print(f"{'-'*50}")
                print(top_segment.get('text', 'No text available'))
                print(f"{'-'*50}")
                
                segment_id = f"{name}_{top_segment.get('segment_id', 0)}"
                all_segments[segment_id] = top_segment
                system_segments = {segment_id: top_segment}
                
                print(f"🧠 Generating answer with {name}...")
                result = self.generate_answer(query, system_segments)
                
                system_results[name] = {
                    "answer": result["answer"],
                    "has_answer": result["has_answer"],
                    "timespan": result["timespan"],
                    "expanded_timespan": result.get("expanded_timespan"),
                    "segments": system_segments
                }
                
                print(f"\n✨ {name} answer: {result['answer']}")
                if result["timespan"]:
                    start, end = result["timespan"]
                    print(f"⏱️ Timespan: {start:.2f}s - {end:.2f}s")
                    
            except Exception as e:
                print(f"⚠️ Error with {name}: {e}")
                import traceback
                traceback.print_exc()
                system_results[name] = {
                    "answer": f"Error processing with {name}: {str(e)}",
                    "has_answer": False,
                    "timespan": None,
                    "segments": {}
                }
        
        return {
            "answer": f"Multiple answers generated - see system_results",
            "has_answer": any(res["has_answer"] for res in system_results.values()),
            "segments": all_segments,
            "system_results": system_results,
            "timespan": None,
        }

    def print_generation_stats(self):
        if not self.generation_log:
            print("No generation statistics available")
            return
        rewrites = [g for g in self.generation_log if g["type"] == "query_rewrite"]
        answers = [g for g in self.generation_log if g["type"] == "answer_generation"]
        print("\n" + "="*80)
        print("📊 GENERATION STATISTICS")
        print("="*80)
        if rewrites:
            print("\n📝 QUERY REWRITES:")
            print(f"Total rewrites: {len(rewrites)}")
            print(f"Model used: {rewrites[0].get('model', 'unknown')}")
            for i, rewrite in enumerate(rewrites):
                print(f"\n{i+1}. Original: \"{rewrite['original']}\"")
                print(f"   Rewritten: \"{rewrite['rewritten']}\"")
        if answers:
            print("\n🔍 ANSWER GENERATIONS:")
            print(f"Total answer generations: {len(answers)}")
            print(f"Model used: {answers[0].get('model', 'unknown')}")
            avg_time = sum(a["gen_time"] for a in answers) / len(answers)
            avg_segments = sum(a["num_segments"] for a in answers) / len(answers)
            avg_context = sum(a["context_length"] for a in answers) / len(answers)
            print(f"Average generation time: {avg_time:.2f}s")
            print(f"Average segments per question: {avg_segments:.1f}")
            print(f"Average context length: {avg_context:.1f} characters")
            for i, answer in enumerate(answers):
                print(f"\n{i+1}. Query: \"{answer['query']}\"")
                print(f"   Answer: \"{answer['answer']}\"")
                print(f"   Generation time: {answer['gen_time']:.2f}s")
        print("\n" + "="*80)