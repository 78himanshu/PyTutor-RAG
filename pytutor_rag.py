from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken
from bs4 import BeautifulSoup
from openai import OpenAI

API_KEY_PATH = Path("./course_api_key.txt")
RAW_DIR = Path("data/raw")
KB_DIR = Path("data/kb")
CHUNKS_PATH = KB_DIR / "chunks.json"
EMBEDS_PATH = KB_DIR / "embeddings.npy"
GENERATION_MODEL = "gpt-5-nano"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 80
TOP_K = 5
DEFAULT_QUERY = "How do Python classes differ from modules, and how should exceptions be handled?"
DOCS = {
    "intro": "https://docs.python.org/3/tutorial/introduction.html",
    "controlflow": "https://docs.python.org/3/tutorial/controlflow.html",
    "datastructures": "https://docs.python.org/3/tutorial/datastructures.html",
    "modules": "https://docs.python.org/3/tutorial/modules.html",
    "inputoutput": "https://docs.python.org/3/tutorial/inputoutput.html",
    "errors": "https://docs.python.org/3/tutorial/errors.html",
    "classes": "https://docs.python.org/3/tutorial/classes.html",
}


def read_api_key() -> str:
    return API_KEY_PATH.read_text(encoding="utf-8").strip()


def tokenizer():
    try:
        return tiktoken.encoding_for_model(GENERATION_MODEL)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def download_once() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in DOCS.items():
        path = RAW_DIR / f"{name}.html"
        if path.exists():
            continue
        print(f"Downloading: {url}")
        with urllib.request.urlopen(url) as resp:
            path.write_bytes(resp.read())


def html_to_text(path: Path) -> tuple[str, str]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else path.stem
    root = soup.find("main") or soup.body or soup
    text = root.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def token_chunks(text: str, enc) -> list[str]:
    ids = enc.encode(text)
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    out = []
    for i in range(0, len(ids), step):
        chunk_ids = ids[i : i + CHUNK_TOKENS]
        if not chunk_ids:
            continue
        chunk = enc.decode(chunk_ids).strip()
        if len(chunk) >= 80:
            out.append(chunk)
    return out


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    rows = []
    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
            encoding_format="float",
        )
        rows.extend(item.embedding for item in resp.data)
    arr = np.array(rows, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def build_kb(client: OpenAI) -> tuple[list[dict], np.ndarray]:
    download_once()
    enc = tokenizer()
    chunks = []
    for name, url in DOCS.items():
        title, text = html_to_text(RAW_DIR / f"{name}.html")
        for idx, chunk in enumerate(token_chunks(text, enc)):
            chunks.append(
                {
                    "source": name,
                    "title": title,
                    "url": url,
                    "chunk_id": idx,
                    "text": chunk,
                }
            )
    print(f"Computing embeddings for {len(chunks)} chunks...")
    embeds = embed_texts(client, [c["text"] for c in chunks])
    KB_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(EMBEDS_PATH, embeds)
    return chunks, embeds


def load_kb(client: OpenAI) -> tuple[list[dict], np.ndarray]:
    if CHUNKS_PATH.exists() and EMBEDS_PATH.exists():
        print("Loading cached knowledge base...")
        chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        embeds = np.load(EMBEDS_PATH)
        return chunks, embeds
    print("Building knowledge base...")
    return build_kb(client)


def retrieve(client: OpenAI, query: str, chunks: list[dict], embeds: np.ndarray) -> list[tuple[float, dict]]:
    q = client.embeddings.create(model=EMBED_MODEL, input=query, encoding_format="float")
    qv = np.array(q.data[0].embedding, dtype=np.float32)
    qv /= np.linalg.norm(qv)
    scores = np.dot(embeds, qv)
    idxs = np.argsort(scores)[-TOP_K:][::-1]
    return [(float(scores[i]), chunks[i]) for i in idxs]


def answer_with_rag(client: OpenAI, query: str, hits: list[tuple[float, dict]]) -> str:
    context = []
    for i, (score, item) in enumerate(hits, start=1):
        context.append(
            f'<retrieved_knowledge id="document-{i}" score="{score:.4f}" source="{item["url"]}">\n'
            f'{item["text"]}\n'
            f'</retrieved_knowledge>'
        )
    response = client.responses.create(
        model=GENERATION_MODEL,
        instructions=(
            "Answer the user's question using only the retrieved knowledge below. "
            "If the retrieved context is insufficient, say so.\n\n" + "\n\n".join(context)
        ),
        input=query,
    )
    if getattr(response, "output_text", None):
        return response.output_text
    parts = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []):
                if getattr(content, "type", None) == "output_text":
                    parts.append(content.text)
    return "\n".join(parts).strip()


def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
    client = OpenAI(api_key=read_api_key())
    chunks, embeds = load_kb(client)
    hits = retrieve(client, query, chunks, embeds)
    answer = answer_with_rag(client, query, hits)

    print(f"API key path: {API_KEY_PATH}")
    print(f"Documents: {len(DOCS)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Query: {query}\n")
    print("Top retrieved chunks:")
    for score, item in hits:
        print(f"- {score:.4f} | {item['source']} | {item['title']}")
    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()