FROM python:3.11-slim
ARG INSTALL_PROMPT_GUARD=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY mock_data ./mock_data

RUN python -m pip install --no-cache-dir . \
    && if [ "$INSTALL_PROMPT_GUARD" = "true" ]; then \
         python -m pip install --no-cache-dir '.[defense]'; \
       fi

RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /app/data/local_db /app/data/gmail /app/data/drive /app/data/orchestrator \
    && chown -R app:app /app

USER app

CMD ["uvicorn", "agent_system.api.agent:app", "--host", "0.0.0.0", "--port", "8000"]
