-- pip install -r requirements.txt
--pip install python-multipart
-- pip freeze > requirements.txt
-- uvicorn api:app --reloacurl -X POST -F "file=@yourfile.pdf" http://127.0.0.1:8000/read-pdf
curl -X POST -H "Content-Type: application/json" \
-d "{\"text\": \"This is a long text to summarize...\"}" \
http://127.0.0.1:8000/summarize

