# Dockerfile
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Default command (will be overridden by docker-compose)
CMD ["echo", "Docker container built! Use docker-compose to start services."]
