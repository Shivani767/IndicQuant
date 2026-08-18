FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY evals ./evals

RUN uv sync --frozen --no-dev --extra docker

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860
CMD ["indicquant", "serve", "--host", "0.0.0.0", "--port", "7860"]
