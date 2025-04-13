#!/bin/bash

# Exit on error
set -e

# Variables - change these as needed
DOCKERHUB_USERNAME="lesterpjy10"
IMAGE_NAME="peract"
TAG="latest"

# Full image name
FULL_IMAGE_NAME="$DOCKERHUB_USERNAME/$IMAGE_NAME:$TAG"

# Build the Docker image
echo "Building Docker image: $FULL_IMAGE_NAME"
docker build -f Dockerfile.peract -t $FULL_IMAGE_NAME .

# Log in to DockerHub (you'll be prompted for password)
echo "Logging in to DockerHub as $DOCKERHUB_USERNAME"
docker login -u $DOCKERHUB_USERNAME

# Push the image to DockerHub
echo "Pushing image to DockerHub: $FULL_IMAGE_NAME"
docker push $FULL_IMAGE_NAME

echo "Done! Image is now available at: $FULL_IMAGE_NAME"