# =============================
# 📘 PDF Reader + Summarizer + Humanizer + Reader
# =============================
# This program can:
# 1️⃣ Read text from a PDF file
# 2️⃣ Summarize the text using AI
# 3️⃣ Read the text out loud using speech
# 4️⃣ Make the text sound more natural ("humanize" it)
# =============================

# Importing libraries we need

import PyPDF2                  # Used to read PDF files and extract text
import pyttsx3                 # Used to convert text into speech
from transformers import pipeline  # Used to access powerful AI models like summarization

# =============================
# 🧠 Load the summarization model
# =============================

# We use the Hugging Face "pipeline" to get a ready-made summarization model.
# The model name "facebook/bart-large-cnn" is a well-known summarizer that can handle long text.
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
)


# =============================
# 📖 Function to read text from a PDF
# =============================
def read_pdf(file_path):
    """Extracts all text from a PDF file and returns it as a string."""

    # Clean up the file path in case the user added quotes (like "C:\path\file.pdf")
    file_path = file_path.strip('"').strip("'")

    # Create a PDF reader object from the file
    reader = PyPDF2.PdfReader(file_path)

    # Initialize an empty string to store all the text from the PDF
    text = ""

    # Loop through every page in the PDF
    for page in reader.pages:
        # Extract text from the current page and add it to our total text
        # Some pages may have no text, so we add an empty string in that case
        text += page.extract_text() or ""

    # Return all the extracted text
    return text


# =============================
# 🗣️ Function to read text out loud
# =============================
def read_aloud(text):
    """Reads text aloud using the pyttsx3 text-to-speech engine."""

    # Initialize the speech engine
    engine = pyttsx3.init()

    # Add the text to be spoken
    engine.say(text)

    # Play the speech and wait until it's done
    engine.runAndWait()


# =============================
# ✍️ Function to summarize long text
# =============================
def summarize_text(text):
    """Uses the AI model to generate a summary of the input text."""

    # The model can only handle text chunks up to a certain size,
    # so we split the text into smaller pieces (about 1000 characters each)
    max_chunk = 1000

    # Create a list of smaller text chunks
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]

    # Initialize an empty string to store all summaries
    summary = ""

    # Go through each chunk and summarize it
    for chunk in chunks:
        # Summarize one chunk
        result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)

        # Extract the summarized text and add it to our summary
        summary += result[0]["summary_text"] + " "

    # Return the complete summary (remove extra spaces)
    return summary.strip()


# =============================
# 💬 Function to humanize text
# =============================
def humanize_text(text):
    """Makes the text sound simpler and more natural."""

    # Here we reuse the summarization model to "simplify" text.
    # We set do_sample=True to make it more creative and varied.
    result = summarizer(text, max_length=200, min_length=50, do_sample=True)

    # Return the humanized (rephrased) text
    return result[0]["summary_text"]


# =============================
# 🏁 Main program logic
# =============================
if __name__ == "__main__":

    # Print a welcome message and show user what options are available
    print("📘 Welcome to the PDF Smart Reader App!")
    print("=========================================")
    print("What would you like to do?")
    print("1️⃣  Read & summarize a PDF")
    print("2️⃣  Read a PDF out loud")
    print("3️⃣  Humanize text manually")
    print("=========================================")

    # Get the user's choice
    choice = input("Enter your choice (1 / 2 / 3): ").strip()

    # ---------------------------
    # 🧩 Option 1: Summarize a PDF
    # ---------------------------
    if choice == "1":
        # Ask for the PDF file path
        file_path = input("Enter PDF file path: ")

        # Tell the user we’re reading the file
        print("\n📖 Reading PDF...")

        # Read text from the PDF file
        text = read_pdf(file_path)

        # Tell the user we’re summarizing the text
        print("\n🧠 Summarizing content...")

        # Summarize the extracted text
        summary = summarize_text(text)

        # Print the summary to the console
        print("\n✅ Summary:\n", summary)

        # Ask the user if they want the app to read the summary out loud
        speak = input("\nDo you want me to read the summary aloud? (y/n): ").lower()

        # If they say yes, read it aloud
        if speak == "y":
            read_aloud(summary)

    # ---------------------------
    # 🎧 Option 2: Read PDF out loud
    # ---------------------------
    elif choice == "2":
        # Ask for the PDF file path
        file_path = input("Enter PDF file path: ")

        # Tell the user we’re reading the file
        print("\n📖 Reading PDF...")

        # Read text from the PDF
        text = read_pdf(file_path)

        # Let the user know it’s being read out loud
        print("\n🎧 Reading the PDF aloud...")

        # Read only the first 2000 characters to prevent super long reads
        read_aloud(text[:2000])

    # ---------------------------
    # 🧍 Option 3: Humanize text manually
    # ---------------------------
    elif choice == "3":
        # Ask the user to input some text
        user_text = input("\n📝 Enter your text: ")

        # Tell the user we’re processing it
        print("\n✨ Making it sound natural...")

        # Use the AI to humanize (rephrase) the text
        humanized = humanize_text(user_text)

        # Show the rephrased text
        print("\n✅ Humanized Text:\n", humanized)

        # Ask if they want it read aloud
        speak = input("\nDo you want me to read it aloud? (y/n): ").lower()

        # If yes, speak the humanized text
        if speak == "y":
            read_aloud(humanized)

    # ---------------------------
    # 🚫 Invalid option handler
    # ---------------------------
    else:
        # Print an error message if the user entered something invalid
        print("❌ Invalid choice! Please run the program again and select 1, 2, or 3.")
