# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Heroku will set PORT dynamically
ENV PORT 8000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY deployment/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/temp \
    && mkdir -p /app/data/uploads \
    && mkdir -p /app/models

# Set execution permissions
RUN chmod +x deployment/startup.sh

# Heroku requires this - don't use EXPOSE
# Heroku automatically exposes ports

# Start application using PORT from environment
CMD ["./deployment/startup.sh"]