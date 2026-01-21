# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies if needed (e.g., for build tools)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy only the dependency files first to leverage Docker cache
COPY pyproject.toml .

# Install dependencies
# We use pip to install the package in editable mode or just dependencies
# Since pyproject.toml is present, we can install direct support
RUN pip install --no-cache-dir .

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "src.server.main"]
