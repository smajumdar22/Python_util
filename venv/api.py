# -----------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------

from fastapi import FastAPI, UploadFile, File      # FastAPI classes for building API & receiving files
from fastapi.responses import JSONResponse         # To return structured JSON responses
import PyPDF2                                      # Library for reading/extracting text from PDFs
import pyttsx3                                     # Text-to-speech engine (optional for API use)
from transformers import pipeline                  # HuggingFace pipeline to load summarization model
import tempfile                                    # Used to store uploaded PDF temporarily

# -----------------------------------------------------------
# INITIALIZE FASTAPI
# -----------------------------------------------------------

app = FastAPI()                                    # Create FastAPI app instance

# -----------------------------------------------------------
# LOAD SUMMARIZATION MODEL
# -----------------------------------------------------------

# Load the "facebook/bart-large-cnn" summarization model using HuggingFace transformers.
# This is done once at startup so it doesn't reload on every request.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------

def read_pdf_from_file(file_path):
    """Extracts all text from a PDF file."""
    
    reader = PyPDF2.PdfReader(file_path)           # Open the PDF using PyPDF2
    text = ""                                      # Container to store extracted text

    for page in reader.pages:                      # Loop through each page
        text += page.extract_text() or ""          # Extract text or add empty string if none

    return text                                    # Return full extracted text



def summarize_text(text):
    """Summarizes long text by breaking it into chunks."""

    max_chunk = 1000                               # Maximum chunk size the model can handle

    # Split the text into smaller pieces so the model can process it
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]

    summary = ""                                   # Store final merged summary

    for chunk in chunks:                           # Process each chunk with the model
        result = summarizer(chunk, 
                            max_length=130, 
                            min_length=30, 
                            do_sample=False)       # Deterministic summarization
        summary += result[0]["summary_text"] + " " # Append chunk summary

    return summary.strip()                         # Clean extra space & return final summary



def humanize_text(text):
    """Makes the text more human and natural sounding."""
    
    result = summarizer(text, 
                        max_length=200, 
                        min_length=50, 
                        do_sample=True)            # do_sample=True makes the output more creative

    return result[0]["summary_text"]               # Return humanized text



def read_aloud(text):
    """Text-to-speech (not used by Angular, but available)."""

    engine = pyttsx3.init()                        # Initialize speech engine
    engine.say(text)                               # Queue the text to speak
    engine.runAndWait()                            # Play the speech audio



# -----------------------------------------------------------
# API ENDPOINTS (USED BY ANGULAR)
# -----------------------------------------------------------

@app.post("/read-pdf")
async def read_pdf_api(file: UploadFile = File(...)):
    """
    Accepts a PDF uploaded from Angular,
    extracts all text, and returns it as JSON.
    """

    # Create a temporary file to store uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())               # Write uploaded data to temp file
        tmp_path = tmp.name                        # Save the file path

    text = read_pdf_from_file(tmp_path)            # Extract text using helper function

    return JSONResponse({"text": text})            # Return extracted text as JSON



@app.post("/summarize")
async def summarize_api(data: dict):
    """
    Accepts raw text from Angular,
    returns a summarized version.
    """

    text = data.get("text", "")                    # Extract 'text' field from JSON body

    if not text:                                   # If no text provided, return error
        return JSONResponse(
            {"error": "No text provided"},
            status_code=400
        )

    summary = summarize_text(text)                 # Call summarization helper

    return JSONResponse({"summary": summary})      # Return summary JSON



@app.post("/humanize")
async def humanize_api(data: dict):
    """
    Accepts text from Angular,
    returns a more natural/rephrased version.
    """

    text = data.get("text", "")                    # Extract text

    if not text:                                   # Validate input
        return JSONResponse(
            {"error": "No text provided"},
            status_code=400
        )

    humanized = humanize_text(text)                # Humanize using model

    return JSONResponse({"humanized": humanized})  # Return JSON response
