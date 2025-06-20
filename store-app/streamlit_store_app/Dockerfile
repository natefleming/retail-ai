FROM python:3.8-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock* ./
COPY src/ ./src/
COPY pages/ ./pages/
COPY app.py ./
COPY data/ ./data/

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Set environment variables
ENV PYTHONPATH=/app

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["poetry", "run", "streamlit", "run", "app.py"] 