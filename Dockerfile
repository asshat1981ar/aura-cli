FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY tools/requirements.txt /tmp/tools-requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/tools-requirements.txt \
    && groupadd --system aura \
    && useradd --system --gid aura --create-home --home-dir /home/aura aura

COPY --chown=aura:aura . /app

USER aura

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "tools.mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
