#!/usr/bin/env python3
"""
Model downloader utility for pre-downloading AI models
"""

import argparse
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Import model configurations
from analyzer import AVAILABLE_MODELS, EMBEDDING_MODELS

def download_llm_model(model_key: str):
    """Download a specific LLM model."""
    if model_key not in AVAILABLE_MODELS:
        print(f"❌ Unknown model: {model_key}")
        print(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        return False
    
    model_config = AVAILABLE_MODELS[model_key]
    model_name = model_config['name']
    
    print(f"📥 Downloading {model_config['description']}...")
    print(f"⚠️  Size: {model_config['size']} - This may take a while...")
    
    try:
        # Download tokenizer
        print("  📝 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Download model
        print("  🧠 Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        print(f"✅ Successfully downloaded {model_key}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_key}: {e}")
        return False

def download_embedding_model(model_key: str):
    """Download a specific embedding model."""
    if model_key not in EMBEDDING_MODELS:
        print(f"❌ Unknown embedding model: {model_key}")
        print(f"Available models: {', '.join(EMBEDDING_MODELS.keys())}")
        return False
    
    model_config = EMBEDDING_MODELS[model_key]
    model_name = model_config['name']
    
    print(f"📥 Downloading {model_config['description']}...")
    print(f"⚠️  Size: {model_config['size']}")
    
    try:
        model = SentenceTransformer(model_name)
        print(f"✅ Successfully downloaded {model_key}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_key}: {e}")
        return False

def download_recommended_set():
    """Download a recommended set of models for different use cases."""
    models_to_download = [
        # Lightweight set
        ("dialogpt", "Lightweight fallback"),
        ("gemma-2b", "Fast general model"),
        ("phi3-mini", "Balanced model"),
        
        # Code-specific (optional)
        ("deepseek-coder", "Best code model (if space allows)"),
    ]
    
    embeddings_to_download = [
        ("all-minilm-l6", "Default embedding"),
        ("code-search", "Code-optimized embedding"),
    ]
    
    print("🎯 Downloading recommended model set...")
    print("This includes models for different speed/quality tradeoffs\n")
    
    success_count = 0
    total_count = 0
    
    # Download embeddings first (smaller)
    for model_key, description in embeddings_to_download:
        print(f"\n📊 {description}")
        if download_embedding_model(model_key):
            success_count += 1
        total_count += 1
    
    # Download LLMs
    for model_key, description in models_to_download:
        print(f"\n🧠 {description}")
        if download_llm_model(model_key):
            success_count += 1
        total_count += 1
    
    print(f"\n📈 Download Summary: {success_count}/{total_count} models successful")
    
    if success_count == total_count:
        print("🎉 All recommended models downloaded successfully!")
    else:
        print("⚠️  Some downloads failed. You can retry individual models later.")

def list_disk_usage():
    """Show estimated disk usage for models."""
    print("💾 Estimated Disk Usage by Model:\n")
    
    print("🧠 Language Models:")
    for key, config in AVAILABLE_MODELS.items():
        print(f"  {key:15} | {config['size']:8} | {config['description']}")
    
    print(f"\n📊 Embedding Models:")
    for key, config in EMBEDDING_MODELS.items():
        print(f"  {key:15} | {config['size']:8} | {config['description']}")
    
    print(f"\n💡 Storage Tips:")
    print(f"  • Models are cached in ~/.cache/huggingface/")
    print(f"  • Docker volume mount preserves downloads between containers")
    print(f"  • Start with phi3-mini + all-minilm-l6 (~7GB total)")
    print(f"  • Add codellama-7b for best code analysis (~13GB extra)")

def main():
    parser = argparse.ArgumentParser(description="Download AI models for offline use")
    parser.add_argument("command", choices=["llm", "embedding", "recommended", "usage"], 
                       help="Type of download")
    parser.add_argument("--model", 
                       help="Specific model to download")
    
    args = parser.parse_args()
    
    if args.command == "usage":
        list_disk_usage()
    
    elif args.command == "recommended":
        download_recommended_set()
    
    elif args.command == "llm":
        if not args.model:
            print("❌ Error: --model required for llm command")
            print(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
            sys.exit(1)
        download_llm_model(args.model)
    
    elif args.command == "embedding":
        if not args.model:
            print("❌ Error: --model required for embedding command")
            print(f"Available models: {', '.join(EMBEDDING_MODELS.keys())}")
            sys.exit(1)
        download_embedding_model(args.model)

if __name__ == "__main__":
    main()