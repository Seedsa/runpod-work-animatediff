# Base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
# FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel
ENV DEBIAN_FRONTEND=noninteractive

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set the working directory
WORKDIR /

# Update and upgrade the system packages (Worker Template)
COPY builder/setup.sh /setup.sh
RUN /bin/bash /setup.sh && \
    rm /setup.sh

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt


# Fetch the model
COPY builder/model_fetcher.py /model_fetcher.py
RUN python3 /model_fetcher.py
RUN rm /model_fetcher.py
# Test
# COPY test_input.json /


# Add src files (Worker Template)
ADD src .

CMD python3 -u /handler.py
