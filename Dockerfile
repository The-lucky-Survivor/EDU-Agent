FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Install system dependencies (as root momentarily if needed, but python slim usually has what we need or we can install pip packages directly)
USER root
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
USER user

# Install pip requirements
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user pysqlite3-binary

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
