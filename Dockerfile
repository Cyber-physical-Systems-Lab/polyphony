FROM ollama/ollama:latest
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    swig \
    libgl1 \
    libglu1-mesa \
    libglut3.12 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    bash \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    "ray[rllib]" \
    torch \
    imageio \
    "gymnasium[all]" \
    opencv-python-headless \
    && python3 -m pip install --break-system-packages --no-cache-dir --upgrade --force-reinstall \
        pyglet==1.5.27
VOLUME /docker-mount/Battery-TA-RWARE
ENTRYPOINT ["/bin/bash"]
