# 1. Base NVIDIA CUDA runtime on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set system-level environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DOCKER_BUILDKIT=1

# Install essential system utilities
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    tar \
    bzip2 \
    ca-certificates \
    git \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libxml2 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Miniconda manually
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# 3. Clone the published DiffPhore repository
WORKDIR /app
ARG CACHEBUST=1
RUN git clone https://github.com/VicFisher/DiffPhore.git

# Shift working directory into the cloned repo root
WORKDIR /app/DiffPhore

# ==========================================
# FIX 1: Fetch and extract the SO3 Cache Data from Zenodo
# ==========================================
# Creates data/cache_diffusion, downloads your archive, and strips the top-level folder 
# so the contents unpack cleanly directly inside data/cache_diffusion/
RUN mkdir -p data/cache_diffusion && \
    wget -q "https://zenodo.org/records/20556749/files/cache_diffusion.tar.gz?download=1" -O /tmp/cache.tar.gz && \
    tar -xzf /tmp/cache.tar.gz -C data/cache_diffusion/ --strip-components=1 && \
    rm /tmp/cache.tar.gz

# ==========================================
# FIX 2: Correct the AncPhore Execution Permissions
# ==========================================
RUN chmod +x ./programs/AncPhore

# 4. Accept Anaconda Terms of Service prior to compilation
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 5. Build the Conda environment using BuildKit package caching
RUN conda env create -f src/environment_docker.yml && \
    conda clean -tiy

# Set default execution shell inside your target conda environment
SHELL ["conda", "run", "-n", "diffphore", "/bin/bash", "-c"]

# 6. Set up the entrypoint execution behavior
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "diffphore", "python", "src/inference.py"]

# Running example in the repo
# docker run --gpus device=0 \
#   -v $(pwd)/diffphore_results:/app/DiffPhore/output_host \
#   diffphore-inference \
#   --phore examples/phore/sQC_QFA_complex.phore \
#   --ligand examples/ligands/STK936575.sdf \
#   --cache_path data/caches \
#   --out_dir output_host \
#   --model_dir weights/diffphore_calibrated_warmuped_ft \
#   --sample_per_complex 40 \
#   --batch_size 20 \
#   --num_workers 6