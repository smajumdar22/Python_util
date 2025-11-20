-- pip install -r requirements.txt
--pip install python-multipart
-- pip freeze > requirements.txt
-- uvicorn api:app --reloacurl -X POST -F "file=@yourfile.pdf" http://127.0.0.1:8000/read-pdf
curl -X POST -H "Content-Type: application/json" \
-d "{\"text\": \"This is a long text to summarize...\"}" \
http://127.0.0.1:8000/summarize

Rendewr build command
pip install -r requirements.txt

Render start command
uvicorn api:app --host 0.0.0.0 --port $PORT


https://python-util-6.onrender.comzjq


