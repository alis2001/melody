#!/bin/bash

# Deploy script for melody container (Direct Docker)
# Usage: ./deploy_melody.sh

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

CONTAINER_NAME="melody_production"
IMAGE_NAME="melody:latest"

echo -e "${BLUE}=== Melody Deployment Script (Docker) ===${NC}"
echo "Time: $(date)"
echo

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating template...${NC}"
    echo "HUGGINGFACE_TOKEN=your_token_here" > .env
    echo -e "${YELLOW}Please edit .env file with your Hugging Face token before continuing.${NC}"
    echo "Press Enter when ready..."
    read
fi

# Check if HUGGINGFACE_TOKEN is set
source .env
if [ -z "$HUGGINGFACE_TOKEN" ] || [ "$HUGGINGFACE_TOKEN" = "your_token_here" ]; then
    echo -e "${RED}❌ HUGGINGFACE_TOKEN not properly set in .env file${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Environment variables configured${NC}"

# Stop and remove existing container
echo -e "\n${YELLOW}🛑 Stopping existing container...${NC}"
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Build new image
echo -e "\n${YELLOW}🔨 Building new image...${NC}"
docker build -t $IMAGE_NAME . --no-cache

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Create required directories
mkdir -p session_reports transcripts default_voices

# Start container with resource limits
echo -e "\n${YELLOW}🚀 Starting container with resource limits...${NC}"
docker run -d \
  --name $CONTAINER_NAME \
  --restart unless-stopped \
  -p 5002:5002 \
  -e HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e MALLOC_TRIM_THRESHOLD_=100000 \
  -e FLASK_ENV=production \
  --memory=4g \
  --memory-reservation=3g \
  --cpus="2.0" \
  --stop-timeout=60 \
  -v $(pwd)/session_reports:/app/session_reports \
  -v $(pwd)/transcripts:/app/transcripts \
  -v $(pwd)/default_voices:/app/default_voices \
  --log-driver json-file \
  --log-opt max-size=50m \
  --log-opt max-file=3 \
  $IMAGE_NAME

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to start container${NC}"
    exit 1
fi

# Wait for container to start
echo -e "\n${YELLOW}⏳ Waiting for container to initialize...${NC}"
sleep 30

# Check if container is running
if docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}" | grep -q "$CONTAINER_NAME"; then
    echo -e "${GREEN}✅ Container started successfully${NC}"
else
    echo -e "${RED}❌ Container failed to start${NC}"
    echo "Container logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Test the application
echo -e "\n${YELLOW}🧪 Testing application...${NC}"
for i in {1..12}; do
    if curl -s -f "http://localhost:5002/refertazione/heartbeat" > /dev/null; then
        echo -e "${GREEN}✅ Application is responding${NC}"
        break
    else
        echo "Waiting for application to respond... ($i/12)"
        sleep 10
    fi
done

# Final health check
if curl -s -f "http://localhost:5002/refertazione/heartbeat" > /dev/null; then
    HEALTH_INFO=$(curl -s "http://localhost:5002/refertazione/heartbeat")
    echo "Application health: $HEALTH_INFO"
else
    echo -e "${YELLOW}⚠️  Application may still be starting up${NC}"
fi

# Show container status
echo -e "\n${YELLOW}📊 Container Status:${NC}"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" $CONTAINER_NAME

# Show recent logs
echo -e "\n${YELLOW}📝 Recent logs:${NC}"
docker logs --tail 15 $CONTAINER_NAME

echo -e "\n${GREEN}🎉 Deployment complete!${NC}"
echo -e "${BLUE}Application available at: http://$(hostname -I | awk '{print $1}'):5002/refertazione/${NC}"
echo -e "${YELLOW}Monitor with: ./monitor_melody.sh${NC}"
echo -e "${YELLOW}View logs with: docker logs -f $CONTAINER_NAME${NC}"
echo -e "${YELLOW}Stop container: docker stop $CONTAINER_NAME${NC}"
