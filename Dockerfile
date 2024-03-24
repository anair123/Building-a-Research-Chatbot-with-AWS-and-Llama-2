# Use the official Python image as base
FROM python:3.9.13

# Set the AWS_DEFAULT_REGION environment variable
ENV AWS_DEFAULT_REGION=us-east-1

# Define build arguments for access key and secret access key
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Set the AWS credentials environment variables
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# Set the working directory inside the container
WORKDIR /app

# upgrade pip and install poetry
RUN pip install --user --no-cache-dir --upgrade pip && \
    pip install --user --no-cache-dir poetry==1.4.2
    
# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install all dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["chainlit", "run", "app.py"]