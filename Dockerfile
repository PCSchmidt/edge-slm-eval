FROM python:3.12-slim AS build
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[llama]"

FROM python:3.12-slim
COPY --from=build /usr/local /usr/local
COPY tests /app/tests
WORKDIR /app
# Run with: docker run --network none -v ./models:/models:ro ...
ENTRYPOINT []
CMD ["edge-slm-eval", "--help"]
