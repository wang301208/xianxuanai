FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY services/skill_gateway /app/skill_gateway

RUN pip install --no-cache-dir \
      fastapi==0.115.0 \
      uvicorn[standard]==0.30.0 \
      httpx==0.27.2

ENV PORT=8300
EXPOSE 8300

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8300') + '/healthz')" || exit 1

CMD ["uvicorn", "skill_gateway.app:app", "--host", "0.0.0.0", "--port", "8300"]
