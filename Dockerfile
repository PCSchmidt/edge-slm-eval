FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY examples ./examples
# Prebuilt CPU wheel of llama-cpp-python: no compiler needed in the image.
RUN pip install --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    -e ".[llama,dev]"
# Offline use: docker run --rm --network none -v "$PWD/models:/models:ro" IMAGE \
#     edge-slm-eval --model /models/model.gguf --out /tmp/results
CMD ["python", "-m", "pytest", "-q"]
