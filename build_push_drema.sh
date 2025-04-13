#!/bin/bash

# Docker Image Build Script
# This script handles building and pushing Docker images for the ML project

# Exit on error
set -e

# Enable BuildKit for more efficient builds
export DOCKER_BUILDKIT=1

# Configuration - replace with your actual Docker Hub username or organization
DOCKER_USERNAME="lesterpjy10"
BASE_IMAGE_NAME="drema-base"
PROJECT_IMAGE_NAME="drema"

# Text formatting
BOLD=$(tput bold)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
BLUE=$(tput setaf 4)
RESET=$(tput sgr0)

# Helper functions
print_section() {
    echo "${BOLD}${BLUE}=== $1 ===${RESET}"
}

print_success() {
    echo "${BOLD}${GREEN}✓ $1${RESET}"
}

print_warning() {
    echo "${BOLD}${YELLOW}! $1${RESET}"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Docker not found. Please install Docker first."
        exit 1
    fi
}

check_login() {
    echo "Checking Docker Hub login status..."
    if ! docker info 2>/dev/null | grep -q "Username"; then
        echo "You are not logged into Docker Hub. Please log in:"
        docker login
    else
        print_success "Already logged into Docker Hub"
    fi
}

build_base_image() {
    print_section "Building Base Image: ${DOCKER_USERNAME}/${BASE_IMAGE_NAME}:latest"
    
    # Check if Dockerfile.base exists
    if [ ! -f "Dockerfile.base" ]; then
        echo "Error: Dockerfile.base not found in current directory!"
        exit 1
    fi
    
    # Build the base image
    echo "Building base image (this may take a while)..."
    docker build -f Dockerfile.base -t ${DOCKER_USERNAME}/${BASE_IMAGE_NAME}:latest .
    
    print_success "Base image built successfully"
    
    # Ask to push the base image
    read -p "Do you want to push the base image to Docker Hub? (y/n): " PUSH_BASE
    if [[ $PUSH_BASE =~ ^[Yy]$ ]]; then
        echo "Pushing base image to Docker Hub..."
        docker push ${DOCKER_USERNAME}/${BASE_IMAGE_NAME}:latest
        print_success "Base image pushed successfully"
    fi
}

build_project_image() {
    print_section "Building Project Image: ${DOCKER_USERNAME}/${PROJECT_IMAGE_NAME}:latest"
    
    # Check if Dockerfile exists
    if [ ! -f "Dockerfile" ]; then
        echo "Error: Dockerfile not found in current directory!"
        exit 1
    fi
    
    # Build the project image
    echo "Building project image..."
    docker build -t ${DOCKER_USERNAME}/${PROJECT_IMAGE_NAME}:latest .
    
    print_success "Project image built successfully"
}

tag_and_push() {
    print_section "Tagging and Pushing Images"
    
    # Get version for tagging
    read -p "Enter version tag for this build (e.g., 1.0.0) or leave empty to skip versioning: " VERSION_TAG
    
    if [ ! -z "$VERSION_TAG" ]; then
        # Remove v prefix if entered
        VERSION_TAG=${VERSION_TAG#v}
        # Tag project image with version
        echo "Tagging project image as ${DOCKER_USERNAME}/${PROJECT_IMAGE_NAME}:v${VERSION_TAG}"
        docker tag ${DOCKER_USERNAME}/${PROJECT_IMAGE_NAME}:latest ${DOCKER_USERNAME}/${PROJECT_IMAGE_NAME}:v${VERSION_TAG}
    else
        print_warning "Skipping version tagging"
    fi
    
    # Ask to push project image
    read -p "Do you want to push the project image to Docker Hub? (y/n): " PUSH_PROJECT
    if [[ $PUSH_PROJECT =~ ^[Yy]$ ]]; then
        echo "Pushing project image (latest) to Docker Hub..."
        docker push ${DOCKER_USERNAME}/${PROJECT_IMAGE_NAME}:latest
        
        if [ ! -z "$VERSION_TAG" ]; then
            echo "Pushing project image (v${VERSION_TAG}) to Docker Hub..."
            docker push ${DOCKER_USERNAME}/${PROJECT_IMAGE_NAME}:v${VERSION_TAG}
        fi
        
        print_success "Project image pushed successfully"
    fi
}

test_images() {
    print_section "Testing Docker Images"
    
    # Ask to test base image
    if [ "$BUILD_BASE" = "true" ]; then
        read -p "Do you want to test the base image? (y/n): " TEST_BASE
        if [[ $TEST_BASE =~ ^[Yy]$ ]]; then
            echo "Starting base image container for testing..."
            echo "Type 'exit' when done testing"
            docker run --rm -it ${DOCKER_USERNAME}/${BASE_IMAGE_NAME}:latest bash
        fi
    fi
    
    # Ask to test project image
    read -p "Do you want to test the project image? (y/n): " TEST_PROJECT
    if [[ $TEST_PROJECT =~ ^[Yy]$ ]]; then
        echo "Starting project image container for testing with mounted volumes..."
        echo "Type 'exit' when done testing"
        docker run --rm -it \
            -v "$(pwd)/data:/local/data" \
            -v "$(pwd)/cache:/local/cache" \
            -v "$(pwd)/work:/local/work" \
            -v "$(pwd)/src:/local/src" \
            -v "$(pwd)/scripts:/local/scripts" \
            -v "$(pwd)/configs:/local/configs" \
            -v "$(pwd)/assets:/local/assets" \
            -v "$(pwd)/drema:/local/drema" \
            -v "$(pwd)/submodules:/local/submodules" \
            --gpus all \
            ${DOCKER_USERNAME}/${PROJECT_IMAGE_NAME}:latest bash
    fi
}

cleanup() {
    print_section "Cleanup Options"
    
    read -p "Do you want to run Docker cleanup to free disk space? (y/n): " DO_CLEANUP
    if [[ $DO_CLEANUP =~ ^[Yy]$ ]]; then
        echo "Running Docker system prune to free disk space..."
        docker system prune -f
        print_success "Cleanup completed"
    fi
}

# Main execution
clear
echo "${BOLD}${BLUE}=========================================${RESET}"
echo "${BOLD}${BLUE}      Project Docker Build Script        ${RESET}"
echo "${BOLD}${BLUE}=========================================${RESET}"

# Check prerequisites
check_docker
check_login

# Ask about building base image
read -p "Do you need to build the base image? (y/n): " BUILD_BASE_REPLY
if [[ $BUILD_BASE_REPLY =~ ^[Yy]$ ]]; then
    BUILD_BASE=true
else
    BUILD_BASE=false
fi

# Execute build process
if [ "$BUILD_BASE" = "true" ]; then
    build_base_image
fi

build_project_image
tag_and_push
test_images
cleanup

print_section "Build Process Complete"
echo "The Docker images have been built and pushed according to your selections."
echo "You can now use these images in your development and deployment workflows."
echo ""
echo "Base Image: ${DOCKER_USERNAME}/${BASE_IMAGE_NAME}:latest"
echo "Project Image: ${DOCKER_USERNAME}/${PROJECT_IMAGE_NAME}:latest"

if [ ! -z "$VERSION_TAG" ]; then
    echo "Tagged Version: ${DOCKER_USERNAME}/${PROJECT_IMAGE_NAME}:v${VERSION_TAG}"
fi

echo "${BOLD}${BLUE}=========================================${RESET}"