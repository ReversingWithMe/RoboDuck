FROM python:3.12.10-slim

WORKDIR /app

# install build tools needed by maturin and quicksect
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl libssl-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Install build deps first so we can compile the Rust extension
RUN python -m pip install --no-cache-dir maturin

COPY . /app

RUN python -m pip install --no-cache-dir -e .

ENTRYPOINT ["python", "stage023_runner.py"]
