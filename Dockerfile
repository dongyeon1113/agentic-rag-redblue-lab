FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /app/data \
    && chown -R app:app /app/data

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app services ./services
COPY --chown=app:app datasets ./datasets

USER app

EXPOSE 8000

CMD ["uvicorn", "services.orchestrator.app:app", "--host", "0.0.0.0", "--port", "8000"]
