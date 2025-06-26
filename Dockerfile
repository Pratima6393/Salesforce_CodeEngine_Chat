# Use official Python base image
FROM python:3.11.9

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all files to the container
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000


# Command to start both FastAPI and Streamlit
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
