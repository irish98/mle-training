
# Base image with Python and necessary libraries
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR ./

# Copy the project files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run the Flask app
CMD [ "python", "src/ingest_data.py" ]