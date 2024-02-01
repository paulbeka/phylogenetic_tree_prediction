#!/bin/sh

# Run your Python script
python3 /app/src/main.py -m data_generation -l src/data/training

# Determine the location from which the Docker image is being run
DOCKER_PARENT_DIR=$(docker inspect --format '{{.HostConfig.Binds}}' $(hostname) | cut -d':' -f2 | cut -d' ' -f1 | head -1)

# Copy the generated file to the determined location
cp /app/src/data_generation_output.pickle "$DOCKER_PARENT_DIR/"

# Optional- keep docker running
tail -f /dev/null