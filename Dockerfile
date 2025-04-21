# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p cache database static templates tmp

# Set environment variables
ENV PORT=8000
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8000

# Run the application
CMD ["sh", "-c", "cd src && python main.py"]
