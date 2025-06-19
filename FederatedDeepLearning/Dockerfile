FROM python:3.12-slim

RUN mkdir /app
WORKDIR /app
COPY script.py ./

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv python install 3.12

# install dependencies
COPY uv.lock /app
COPY pyproject.toml /app
RUN uv sync --locked

ENV PATH="/app/.venv/bin:$PATH"
RUN chmod -R go+rX /app

CMD ["python", "script.py"]