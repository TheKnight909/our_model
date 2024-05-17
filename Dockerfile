# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Copy requirements.txt separately to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies first to leverage Docker cache
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
