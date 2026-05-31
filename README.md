# PyTutor-RAG

A Python-based Retrieval-Augmented Generation (RAG) system that builds a searchable knowledge base from official Python documentation and answers user questions using semantic retrieval.

## Overview

PyTutor-RAG downloads selected sections of the official Python tutorial, chunks the documents using token-based semantic chunking, generates embeddings for each chunk, and stores the knowledge base locally. It then performs semantic similarity search over the stored embeddings and sends the most relevant retrieved context to the OpenAI Responses API to generate grounded answers.

The project demonstrates the full RAG workflow: document ingestion, chunking, embedding generation, vector retrieval, and retrieval-augmented response generation.

## What This Project Does

The system performs the following steps:

1. Downloads Python documentation pages from official sources
2. Extracts readable text from HTML documents
3. Splits the text into token-based chunks
4. Generates embeddings for each chunk
5. Saves chunks and embeddings locally
6. Loads the cached knowledge base on future runs
7. Retrieves the most relevant chunks for a query
8. Sends retrieved context to the Responses API
9. Generates an answer grounded in the retrieved documentation

## Documents Used

The knowledge base is built from the following Python tutorial pages:

- Introduction
- More Control Flow Tools
- Data Structures
- Modules
- Input and Output
- Errors and Exceptions
- Classes

## Features

- Automated document downloading
- HTML text extraction using BeautifulSoup
- Token-based chunking with overlap
- OpenAI embedding generation
- Local vector storage using NumPy
- Cached knowledge base to avoid repeated embedding computation
- Semantic similarity search using cosine similarity
- Retrieval-augmented answer generation using the OpenAI Responses API
- Lightweight implementation without SQLite or external vector database wrappers

## Tech Stack

- Python
- OpenAI API
- BeautifulSoup
- NumPy
- Tiktoken
- JSON
- urllib

## Project Structure

```text
PyTutor-RAG/
│
├── pytutor_rag.py
├── requirements.txt
├── pipeline_execution.png
├── .gitignore
└── README.md
```

Generated output after running the script:

```text
data/
│
├── raw/
│   ├── intro.html
│   ├── controlflow.html
│   ├── datastructures.html
│   ├── modules.html
│   ├── inputoutput.html
│   ├── errors.html
│   └── classes.html
│
└── kb/
    ├── chunks.json
    └── embeddings.npy
```

## Installation

Clone the repository:

```bash
git clone https://github.com/78himanshu/PyTutor-RAG.git
cd PyTutor-RAG
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a local API key file in the project root:

```text
course_api_key.txt
```

Add your OpenAI API key inside that file.

The API key file is intentionally excluded from GitHub using `.gitignore`.

## Usage

Run the script with the default query:

```bash
python pytutor_rag.py
```

Or pass your own query:

```bash
python pytutor_rag.py "How do Python classes differ from modules?"
```

## Example Output

```text
Building knowledge base...
Computing embeddings for 159 chunks...
API key path: course_api_key.txt
Documents: 7
Chunks: 159
Query: How do Python classes differ from modules, and how should exceptions be handled?

Top retrieved chunks:
- 0.5649 | classes | 9. Classes — Python documentation
- 0.4975 | classes | 9. Classes — Python documentation
- 0.4902 | errors | 8. Errors and Exceptions — Python documentation
```

## Why This Matters

RAG systems improve the reliability of AI answers by grounding responses in retrieved source material. This project demonstrates how a small, focused knowledge base can be built from technical documentation and used to answer programming questions with relevant supporting context.

## Future Improvements

- Add a command-line interface for custom configuration
- Add source citations in the generated answer
- Support additional documentation collections
- Add retrieval evaluation metrics
- Add a web interface for interactive Q&A
- Package the system as a reusable Python module

## Author

Himanshu Paithane

Master of Science in Computer Science  
Stevens Institute of Technology
