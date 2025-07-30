FROM python:3.10-slim

WORKDIR /app

# Install build dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your backend code
COPY . .

# If your entry point is main.py and your FastAPI app is called "app" (inside main.py)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
