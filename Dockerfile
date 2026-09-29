# Streamlit chat demo with the precomputed taste artifacts baked in.
#   docker build -t trusted-ai .
#   docker run --rm -p 8501:8501 --env-file .env trusted-ai
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# data/score_model/gbm.pkl was pickled with scikit-learn 1.8.0 (see data/score_model/meta.json)
COPY requirements.txt .
RUN pip install -r requirements.txt "scikit-learn==1.8.*"

COPY trusted_ai/ trusted_ai/
COPY app/ app/
COPY evaluation/ evaluation/
COPY scripts/ scripts/
COPY data/ data/
COPY README.md REPORT.md ./

# Conversation logs and new query embeddings are written at runtime
RUN useradd --create-home appuser \
    && mkdir -p logs/conversations data/embedding_cache/queries \
    && chown -R appuser:appuser logs data/embedding_cache
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
