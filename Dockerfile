# Local dev/testing parity only -- Streamlit Community Cloud (Part E) is
# the actual deployment target and builds directly from the repo via
# requirements.txt, not this image.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
