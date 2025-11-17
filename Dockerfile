# ---- Base stage ----
FROM huggingface/transformers-pytorch-gpu:latest

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for Poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user with sudo privileges
RUN useradd -m -s /bin/bash gabor \
    && echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN pip install datasets numpy pandas scikit-learn tqdm matplotlib seaborn kaggle black jupyter ipykernel 
# Set working directory for that user
RUN chown -R gabor /home/gabor 

# Switch to the new user
USER gabor
RUN chmod -R 777 /home/gabor

WORKDIR /home/gabor/app
COPY . /home/gabor/app

EXPOSE 8888

# ---- Default command ----
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
