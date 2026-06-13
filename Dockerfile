FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY configs ./configs
COPY data/processed ./data/processed
RUN mkdir -p /app/outputs /app/models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
CMD ["uvicorn", "agentic_eval_framework.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
