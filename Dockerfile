FROM python:3.11-slim

# Install system dependencies (ffmpeg required for audio processing via pydub)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create the persistent data directory for SQLite
# Mount a volume here in production: DB_PATH=/data/synthnotes.db
RUN mkdir -p /data

# Default to 8501; Railway overrides PORT at runtime with its own dynamic value.
# Shell form (not exec form) is used so ${PORT} is expanded correctly.
ENV PORT=8501
EXPOSE ${PORT}

CMD streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
