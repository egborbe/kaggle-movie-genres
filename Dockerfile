# ---- Base stage ----
FROM huggingface/transformers-pytorch-gpu:latest

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for Poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# ---- Install Poetry ----
ENV POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# ---- Configure Poetry ----
# 1. Keep Poetry’s venv inside project folder (.venv)
# 2. Don’t ask interactive questions
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_NO_INTERACTION=1

# ---- Working directory ----
WORKDIR /app

# ---- Copy project files ----
COPY pyproject.toml poetry.lock* /app/

# ---- Install dependencies (no root packages yet) ----
RUN poetry install --no-root --no-dev

# ---- Copy the rest of the app ----
COPY . /app

# ---- Default command ----
CMD ["poetry", "run", "python", "main.py"]

