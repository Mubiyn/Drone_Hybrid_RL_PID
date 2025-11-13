# Dockerfile for Drone Hybrid RL+PID Training
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install project in editable mode
RUN pip install -e .

# Expose TensorBoard port
EXPOSE 6006

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command
CMD ["bash"]
