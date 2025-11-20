# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import PyPDF2
from transformers import pipeline
import tempfile

# -----------------------------------------------------------
# INITIALIZE FASTAPI
# -----------------------------------------------------------

app = FastAPI()

# -----------------------------------------------------------
# LOAD SUMMARIZATION MODEL
# -----------------------------------------------------------

# Load summarization model once at startup
summarizer = pipeline(
    "summarization",
    model="philschmid/tiny-t5-summarization"
)


# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------

def read_pdf_from_file(file_path):
    """Extracts all text from a PDF file."""
    reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def summarize_text(text):
    """Summarizes long text by breaking it into chunks."""
    max_chunk = 1000
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
    summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summary += result[0]["summary_text"] + " "
    return summary.strip()


def humanize_text(text):
    """Makes the text more human and natural sounding."""
    result = summarizer(text, max_length=200, min_length=50, do_sample=True)
    return result[0]["summary_text"]


def read_aloud(text):
    """Text-to-speech disabled for cloud deployment."""
    # pyttsx3 is Windows-only and won't work on Linux servers
    # Use gTTS or other cloud TTS if needed
    return None

# -----------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------

@app.post("/read-pdf")
async def read_pdf_api(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    text = read_pdf_from_file(tmp_path)
    return JSONResponse({"text": text})


@app.post("/summarize")
async def summarize_api(data: dict):
    text = data.get("text", "")
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    summary = summarize_text(text)
    return JSONResponse({"summary": summary})


@app.post("/humanize")
async def humanize_api(data: dict):
    text = data.get("text", "")
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    humanized = humanize_text(text)
    return JSONResponse({"humanized": humanized})
