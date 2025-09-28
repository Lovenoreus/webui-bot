#!/usr/bin/env python3
import sys
import traceback
import logging
from datetime import datetime
import os
import json
import time
import re
import requests
from typing import List, Dict, Generator
from pprint import pprint
from tqdm import tqdm
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# ==============================
# CONFIG FLAG
# ==============================
USE_OPENAI = False  # <--- toggle here
# ==============================

# --- Load .env if OpenAI mode ---
if USE_OPENAI:
    load_dotenv()
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# MODEL CONFIG
# -------------------------------
if USE_OPENAI:
    EMBEDDINGS_MODEL_NAME = "text-embedding-3-large"
    CHUNK_ENRICHMENT_MODEL = "gpt-4o-mini"
else:
    OLLAMA_HOST = "vs2153.vll.se"
    OLLAMA_PORT = 11434
    EMBEDDINGS_MODEL_NAME = "jeffh/intfloat-multilingual-e5-large:q8_0"
    CHUNK_ENRICHMENT_MODEL = "gpt-oss:20b"

DEBUG = True

# -------------------------------
# UTILS
# -------------------------------
def clean_text(s: str) -> str:
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", s).strip()

def clean_json_response(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    return cleaned.strip()

# -------------------------------
# EMBEDDING FUNCTIONS
# -------------------------------
def embed_with_openai(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings with OpenAI API
    """
    resp = openai_client.embeddings.create(
        model=EMBEDDINGS_MODEL_NAME,
        input=texts
    )
    return [d.embedding for d in resp.data]

def embed_with_ollama(texts: List[str], model_name: str, ollama_url: str) -> Generator[List[float], None, None]:
    def get_single_embedding(text: str) -> List[float]:
        payload = {"model": model_name, "prompt": text}
        response = requests.post(ollama_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["embedding"]

    for text in texts:
        try:
            yield get_single_embedding(text)
        except Exception as e:
            print(f"Embedding failed: {e}")
            yield [0.0] * 768

# -------------------------------
# LLM ENRICHMENT
# -------------------------------
def enrich_chunk_with_llm(input_chunk: dict) -> dict:
    prompt = f"""
    You are preparing a JSON chunk for Retrieval-Augmented Generation (RAG).

    Input:
    file_name: {input_chunk.get('file_name')}
    section_id: {input_chunk.get('section_id')}
    level: {input_chunk.get('level')}
    title: {input_chunk.get('title')}
    text: {input_chunk.get('text')}

    Task:
    - Generate a JSON object with the following keys:
        "source_file": (only filename)
        "section": (input title)
        "page": 1
        "section_id": (from input)
        "main_section_id": (from input level)
        "context_summary": (short Swedish summary)
        "keywords": (list of Swedish keywords)
        "content_type": ("narrative", "instruction", "reference")
        "hierarchical_tags": (list of Swedish tags)
        "clinical_domain": (list of Swedish domains)
        "text": (copy input text)
    Output: ONLY a valid JSON object
    """

    if USE_OPENAI:
        resp = openai_client.chat.completions.create(
            model=CHUNK_ENRICHMENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw_output = resp.choices[0].message.content.strip()
    else:
        url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
        payload = {"model": CHUNK_ENRICHMENT_MODEL, "prompt": prompt, "stream": False}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        raw_output = response.json().get("response", "").strip()

    cleaned_output = clean_json_response(raw_output)
    return json.loads(cleaned_output)

# -------------------------------
# PDF HELPERS
# -------------------------------
SECTION_PATTERN = re.compile(r"""(?mx)^(?P<sid>\d+\.\d+(?:\.\d+)*\.?)[ \t]+(?=[A-ZÅÄÖ])""")
TITLE_CAPTURE_PATTERN = re.compile(r"""(?mx)^(?P<sid>\d+\.\d+(?:\.\d+)*\.?)[ \t]+(?P<title>[^\n\r]+)""")
TOC_LEADER_RE   = re.compile(r'[._\-•]{3,}')
TOC_PAGENO_RE   = re.compile(r'\s\d{1,4}\s*$')

def looks_like_toc_title(title: str) -> bool:
    return bool(TOC_LEADER_RE.search(title.strip()) or TOC_PAGENO_RE.search(title.strip()))

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def find_sections(raw_text: str) -> List[Dict]:
    sections = []
    for m in SECTION_PATTERN.finditer(raw_text):
        sid = m.group("sid")
        line_start = raw_text.rfind("\n", 0, m.start()) + 1
        line_end_n = raw_text.find("\n", m.start())
        line = raw_text[line_start:line_end_n if line_end_n != -1 else len(raw_text)]
        title_match = TITLE_CAPTURE_PATTERN.match(line)
        title = title_match.group("title").strip() if title_match else ""
        if looks_like_toc_title(title):
            continue
        sections.append({"sid": sid, "pos": m.start(), "title": title})
    return sections

def build_raw_chunks(raw_text: str, file_name: str) -> List[Dict]:
    secs = find_sections(raw_text)
    chunks = []
    for i, sec in enumerate(secs):
        start, end = sec["pos"], secs[i + 1]["pos"] if i + 1 < len(secs) else len(raw_text)
        block = raw_text[start:end]
        first_newline = block.find("\n")
        body = "" if first_newline == -1 else block[first_newline + 1 :]
        tm = TITLE_CAPTURE_PATTERN.match(block[:first_newline] if first_newline != -1 else block)
        sid = (tm.group("sid") if tm else sec["sid"]).rstrip(".")
        title = (tm.group("title") if tm else sec["title"]).strip()
        chunk_text = clean_text(body)
        if not chunk_text:
            continue
        chunks.append({
            "file_name": file_name,
            "section_id": sid,
            "level": sid.count(".") + 1,
            "title": title,
            "text": chunk_text
        })
    return chunks

# -------------------------------
# MAIN
# -------------------------------
def get_list_of_files(src_dir):
    return [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".pdf")]

def main():
    src_dir = r"C:\\RemoteRepos\\chatbot_onprem\\cosmic_documents_43"
    if len(sys.argv) > 1:
        src_dir = sys.argv[1]
    else:
        print("Usage: python llm_chunking_by_id_ollama_openai.py <source_directory>")
        sys.exit(1)
    file_paths = get_list_of_files(src_dir)
    all_chunks = []

    print("Extracting chunks...")
    for file_path in tqdm(file_paths, desc="PDFs", unit="file"):
        file_name = os.path.basename(file_path)
        raw = read_pdf_text(file_path)
        all_chunks.extend(build_raw_chunks(raw, file_name))

    print(f"Total chunks: {len(all_chunks)}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, f"enriched_chunks_{timestamp}.json")

    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write("[\n")
        first = True
        for idx, chunk in enumerate(tqdm(all_chunks, desc="Enriching", unit="chunk")):
            try:
                enriched = enrich_chunk_with_llm(chunk)
                if not first:
                    f_out.write(",\n")
                json.dump(enriched, f_out, ensure_ascii=False, indent=2)
                first = False
            except Exception:
                print(traceback.format_exc())
        f_out.write("\n]\n")

    print(f"Saved enriched chunks to {output_file}")

if __name__ == "__main__":
    main()
