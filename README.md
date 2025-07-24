# 🐳 Multi-Model Codebase AI Analyzer

A powerful, offline AI system that can read and analyze any codebase using **multiple state-of-the-art models**. Choose from 8+ language models including **CodeLlama**, **DeepSeek Coder**, **Phi-3**, and more - all running completely offline!

## ✨ Key Features

- **🤖 Multiple AI Models**: 8+ LLMs including code-specialized models
- **📊 Dynamic Model Selection**: Choose models based on speed vs quality needs
- **💻 Code-Optimized**: CodeLlama, DeepSeek Coder for superior code understanding
- **⚡ Performance Options**: From ultra-fast (Gemma 2B) to high-quality (Llama3 8B)
- **🔍 Smart Embeddings**: Multiple embedding models optimized for code search
- **🐳 Fully Dockerized**: Zero setup issues, runs anywhere
- **💾 Persistent Cache**: Downloaded models persist between runs
- **🎯 RAG-Powered**: Retrieval-Augmented Generation for precise answers

## 🧠 Available Models

### 💻 Code-Specialized Models (Recommended for Programming)
- **codellama-7b** ⭐⭐⭐⭐⭐ - Meta's CodeLlama, best for code analysis (~13GB)
- **deepseek-coder** ⭐⭐⭐⭐⭐ - DeepSeek Coder, specialized for programming (~12GB)  
- **codeqwen-7b** ⭐⭐⭐⭐⭐ - Alibaba's CodeQwen, excellent code understanding (~13GB)

### 🧠 General-Purpose Models  
- **phi3-mini** ⭐⭐⭐⭐ - Microsoft Phi-3, fast and capable (~7GB)
- **mistral-7b** ⭐⭐⭐⭐ - Mistral 7B, great balance (~13GB)
- **llama3-8b** ⭐⭐⭐⭐⭐ - Meta Llama 3, high quality (~16GB)
- **gemma-2b** ⭐⭐⭐ - Google Gemma, very fast (~5GB)

### 🔍 Embedding Models
- **code-search**: Optimized for code search and analysis
- **all-minilm-l6**: Fast and efficient, good balance
- **all-minilm-l12**: Better quality, slightly slower

## 🚀 Quick Start

### 1. One-Command Setup
```bash
chmod +x docker-setup.sh && ./docker-setup.sh
```

### 2. See Available Models
```bash
docker compose run --rm codebase-analyzer python3 analyzer.py list-models
```

### 3. Choose Your Model & Start Analyzing
```bash
# Fast analysis (5GB model)
docker compose run --rm codebase-analyzer python3 analyzer.py interactive --path /app/codebase --llm gemma-2b

# Best for code (13GB model)  
docker compose run --rm codebase-analyzer python3 analyzer.py interactive --path /app/codebase --llm codellama-7b --embedding code-search

# Balanced option (7GB model)
docker compose run --rm codebase-analyzer python3 analyzer.py interactive --path /app/codebase --llm phi3-mini
```

## 📋 Commands & Usage

### List Available Models
```bash
docker compose run --rm codebase-analyzer python3 analyzer.py list-models
```

### Pre-download Models (Optional)
```bash
# Download recommended model set
docker compose run --rm codebase-analyzer python3 download_models.py recommended

# Download specific models
docker compose run --rm codebase-analyzer python3 download_models.py llm --model codellama-7b
docker compose run --rm codebase-analyzer python3 download_models.py embedding --model code-search

# Check disk usage
docker compose run --rm codebase-analyzer python3 download_models.py usage
```

### Index Your Codebase
```bash
# Index with default model
docker compose run --rm codebase-analyzer python3 analyzer.py index --path /app/codebase

# Index with specific model (embedding model affects search)
docker compose run --rm codebase-analyzer python3 analyzer.py index --path /app/codebase --embedding code-search
```

### Interactive Analysis
```bash
# Quick presets
docker compose up fast-analyzer      # Gemma 2B (fastest)
docker compose up code-analyzer      # CodeLlama 7B (best for code)

# Custom model selection
docker compose run --rm codebase-analyzer python3 analyzer.py interactive \
  --path /app/codebase \
  --llm deepseek-coder \
  --embedding code-search
```

### Single Questions
```bash
docker compose run --rm codebase-analyzer python3 analyzer.py query \
  --path /app/codebase \
  --llm phi3-mini \
  --question "How does the authentication system work?"
```

## 🎯 Example Questions You Can Ask

- **Architecture**: "How is the authentication system structured?"
- **Functions**: "Show me all the database connection functions"
- **Bugs**: "Are there any potential memory leaks in this code?"
- **Dependencies**: "What external libraries does this project use?"
- **Patterns**: "What design patterns are implemented here?"
- **Security**: "Are there any security vulnerabilities?"
- **Performance**: "Which functions might be performance bottlenecks?"
- **Documentation**: "Explain how the API endpoints work"

## 📁 Supported File Types

```
Programming Languages:
.py .js .ts .java .cpp .c .h .hpp .cs .rb .go .rs .php .swift .kt
.scala .clj .hs .ml .r .sql .sh

Configuration & Data:
.yaml .yml .json .xml .html .css .md .txt .dockerfile .makefile
```

## 🔧 Configuration

### Using Your Own Codebase

**Method 1: Edit docker compose.yml**
```yaml
volumes:
  - /path/to/your/project:/app/codebase:ro  # Change this path
```

**Method 2: Copy to example directory**
```bash
# Copy your code to the example directory
cp -r /path/to/your/project/* ./example_codebase/
```

### Memory Usage
- **Default**: ~2-4GB RAM
- **Container Limits**: Add to docker compose.yml:
```yaml
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G
```

### GPU Support (Optional)
Uncomment GPU section in docker compose.yml:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Model Customization
Modify the Dockerfile to use different models:
```dockerfile
# For better quality (needs more RAM):
RUN python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-large')
"
```

## 📊 Performance

| Codebase Size | Indexing Time | Query Time | Memory Usage |
|---------------|---------------|------------|--------------|
| Small (< 1K files) | 1-2 minutes | < 5 seconds | 2GB |
| Medium (1K-10K files) | 5-15 minutes | < 10 seconds | 3GB |
| Large (10K+ files) | 15-60 minutes | < 15 seconds | 4GB+ |

## 🛡️ Privacy & Security

- **Completely Offline**: Your code never leaves your machine
- **No Telemetry**: No data collection or external calls
- **Local Storage**: All embeddings stored in `./chroma_db/`
- **Open Source**: Full transparency, modify as needed

## 🚨 Troubleshooting

### Common Issues

**"Docker not found"**
Install Docker from: https://docs.docker.com/get-docker/

**"Permission denied"**
```bash
# Make sure you have Docker permissions
sudo usermod -aG docker $USER
# Log out and back in, or:
newgrp docker
```

**"Container runs out of memory"**
```bash
# Increase Docker memory limit in Docker Desktop settings
# Or add memory limits to docker compose.yml
```

**"No relevant code found"**
- Try rephrasing your question
- Ensure the codebase was indexed first
- Check if your files are in supported formats
- Verify volume mounts in docker compose.yml

**"Models downloading slowly"**
- Models are cached after first download
- Total download size: ~1.5GB
- Use `docker compose build --no-cache` to re-download

### Performance Tuning

**Faster indexing**:
Edit analyzer.py:
```python
chunk_size = 25  # Default: 50
```

**Better accuracy**:
Edit analyzer.py:
```python
relevant_chunks = self.search_code(question, top_k=5)  # Default: 3
```

**Container optimization**:
```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1
docker compose build
```

## 🔄 Updates & Maintenance

### Re-indexing
Re-run indexing when your codebase changes significantly:
```bash
docker compose run --rm codebase-analyzer python3 analyzer.py index --path /app/codebase
```

### Database Management
Vector database persisted in `./chroma_db/`:
```bash
# Reset database
rm -rf chroma_db/
docker compose run --rm codebase-analyzer python3 analyzer.py index --path /app/codebase

# Backup database
tar -czf chroma_backup.tar.gz chroma_db/

# Restore database
tar -xzf chroma_backup.tar.gz
```

### Container Updates
```bash
# Update to latest base image
docker compose build --pull --no-cache

# Clean up old images
docker image prune -f
```

## 🤝 Contributing

1. Fork the repository
2. Make your changes
3. Test with different codebases
4. Submit a pull request

## 📄 License

MIT License - Use freely for personal and commercial projects.

## 🆘 Support

- Check troubleshooting section above
- Review code comments for implementation details
- Open GitHub issues for bugs or feature requests

---

**Made with ❤️ for developers who value privacy and precision**