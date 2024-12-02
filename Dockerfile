# Use the specified base image
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Set the working directory
WORKDIR /video-translation

# Clean the apt cache
RUN apt clean

# Update the package list
RUN apt update -y
RUN apt upgrade -y

# Install ffmpeg
RUN apt install -y ffmpeg

# Copy the requirements file to the container's working directory
COPY requirements.txt .

# Install the requirements
RUN pip install -r requirements.txt

# Copy all files to the container's working directory
COPY . .

# Run a jupyter server as the default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
