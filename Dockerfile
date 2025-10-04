# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy Python dependencies first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command to run training
CMD ["python", "llm_train.py"]
