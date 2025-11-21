# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import PyPDF2
import tempfile
import requests
import os

# -----------------------------------------------------------
# INITIALIZE FASTAPI
# -----------------------------------------------------------

app = FastAPI(title="PDF Summarizer API")

# Enable CORS so frontend can access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# HUGGING FACE INFERENCE API CONFIG
# -----------------------------------------------------------

HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # Set your token in environment
HF_API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def hf_summarize(text: str) -> str:
    """Send text to Hugging Face Inference API and get the summary."""
    payload = {"inputs": text}
    try:
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and "summary_text" in data[0]:
            return data[0]["summary_text"]
        return str(data)
    except requests.RequestException as e:
        return f"Error calling Hugging Face API: {str(e)}"

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------

def read_pdf_from_file(file_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text: str, max_length: int = 1000, overlap: int = 100):
    """
    Split text into overlapping chunks.
    - max_length: approx number of characters per chunk
    - overlap: number of characters overlapping between chunks
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunks.append(text[start:end])
        start += max_length - overlap
    return chunks


def summarize_long_text(text: str) -> str:
    """Summarize long text in chunks via Hugging Face API."""
    chunks = chunk_text(text, max_length=1000, overlap=100)
    summaries = []
    for chunk in chunks:
        summary = hf_summarize(chunk)
        summaries.append(summary)
    return " ".join(summaries)

# -----------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "PDF Summarizer API is running!"}


@app.post("/read-pdf")
async def read_pdf_api(file: UploadFile = File(...)):
    """Extract text from uploaded PDF."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        text = read_pdf_from_file(tmp_path)
        return JSONResponse({"text": text})
    except Exception as e:
        return JSONResponse({"error": f"Failed to read PDF: {str(e)}"}, status_code=500)


@app.post("/summarize")
async def summarize_api(data: dict):
    """Summarize text using Hugging Face API."""
    text = data.get("text", "")
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    summary = summarize_long_text(text)
    return JSONResponse({"summary": summary})


@app.post("/humanize")
async def humanize_api(data: dict):
    """Humanize / rephrase text using Hugging Face API."""
    text = data.get("text", "")
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    humanized = summarize_long_text(text)
    return JSONResponse({"humanized": humanized})
