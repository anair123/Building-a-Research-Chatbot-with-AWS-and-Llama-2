# Use the official Python image as base
FROM python:3.9.13

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install all dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Command to run the application
CMD ["chainlit", "run", "app.py", "--host=0.0.0.0", "--port=80", "--headless"]