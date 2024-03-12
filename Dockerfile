# Use the official Python image as base
FROM python:3.9.13

# Set the working directory inside the container
WORKDIR /app

# install poetry
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir poetry==1.4.2
    
# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install all dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

EXPOSE 8000

# Command to run the application
CMD ["chainlit", "run", "app.py"]