FROM ollama/ollama:latest
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    swig \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    bash \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --break-system-packages "ray[rllib]" torch imageio
RUN python3 -m pip install --break-system-packages "gymnasium[all]"
RUN python3 -m pip install --break-system-packages opencv-python-headless
VOLUME /docker-mount/Battery-TA-RWARE
ENTRYPOINT ["/bin/bash"]
