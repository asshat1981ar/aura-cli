# Use a lightweight Python image as the base
FROM python:3.12-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY ./tools/requirements.txt /app/tools/requirements.txt
RUN pip install --no-cache-dir -r /app/tools/requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Set the GITHUB_PAT environment variable securely (e.g., via Kubernetes Secret or Docker secret)
# For local testing, you can pass it via -e GITHUB_PAT=YOUR_PAT during docker run

# Command to run the Uvicorn server
# Make sure to run the mcp_server.py from the tools directory
CMD ["uvicorn", "tools.mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
