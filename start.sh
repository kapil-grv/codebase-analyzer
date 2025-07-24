#!/bin/bash

echo "🚀 Starting Codebase AI Analyzer Web UI..."

# Build the Docker image
echo "📦 Building Docker image..."
docker compose build

# Start the services
echo "🌐 Starting web UI..."
docker compose up -d

echo "✅ Web UI is starting up!"
echo "🌍 Open your browser and go to: http://localhost:8080"
echo ""
echo "📋 Available commands:"
echo "  docker compose logs -f    # View logs"
echo "  docker compose down       # Stop services"
echo "  docker compose restart    # Restart services" 