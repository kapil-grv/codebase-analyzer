#!/usr/bin/env python3
"""
Offline Codebase AI Analyzer
A simple system to analyze codebases and answer questions using local AI models.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
import hashlib

# Core dependencies
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Available models configuration
AVAILABLE_MODELS = {
    # Code-specific models (recommended)
    "codellama-7b": {
        "name": "codellama/CodeLlama-7b-Instruct-hf",
        "description": "Meta's CodeLlama 7B - Best for code analysis",
        "size": "~13GB",
        "quality": "⭐⭐⭐⭐⭐",
        "speed": "Medium",
        "type": "code"
    },
    "codeqwen-7b": {
        "name": "Qwen/CodeQwen1.5-7B-Chat",
        "description": "Alibaba's CodeQwen - Excellent for code understanding",
        "size": "~13GB", 
        "quality": "⭐⭐⭐⭐⭐",
        "speed": "Medium",
        "type": "code"
    },
    "deepseek-coder": {
        "name": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "description": "DeepSeek Coder - Specialized for programming",
        "size": "~12GB",
        "quality": "⭐⭐⭐⭐⭐",
        "speed": "Medium",
        "type": "code"
    },
    
    # Smaller, faster models
    "phi3-mini": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "description": "Microsoft Phi-3 Mini - Fast and capable",
        "size": "~7GB",
        "quality": "⭐⭐⭐⭐",
        "speed": "Fast",
        "type": "general"
    },
    "gemma-2b": {
        "name": "google/gemma-2b-it",
        "description": "Google Gemma 2B - Very fast, decent quality (requires HF login)",
        "size": "~5GB",
        "quality": "⭐⭐⭐",
        "speed": "Very Fast",
        "type": "general",
        "requires_auth": True
    },
    
    # Larger, higher quality models
    "llama3-8b": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "description": "Meta Llama 3 8B - High quality general model",
        "size": "~16GB",
        "quality": "⭐⭐⭐⭐⭐",
        "speed": "Slow",
        "type": "general"
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Mistral 7B - Great balance of speed and quality",
        "size": "~13GB",
        "quality": "⭐⭐⭐⭐",
        "speed": "Medium",
        "type": "general"
    },
    
    # Legacy/lightweight options
    "dialogpt": {
        "name": "microsoft/DialoGPT-medium",
        "description": "DialoGPT - Lightweight but limited for code",
        "size": "~1GB",
        "quality": "⭐⭐",
        "speed": "Very Fast",
        "type": "legacy"
    }
}

# Embedding models
EMBEDDING_MODELS = {
    "all-minilm-l6": {
        "name": "all-MiniLM-L6-v2",
        "description": "Fast and efficient - Good balance",
        "size": "~90MB",
        "dimensions": 384
    },
    "all-minilm-l12": {
        "name": "all-MiniLM-L12-v2", 
        "description": "Better quality, slower",
        "size": "~130MB",
        "dimensions": 384
    },
    "code-search": {
        "name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
        "description": "Optimized for code search",
        "size": "~90MB",
        "dimensions": 384
    }
}

class CodebaseAnalyzer:
    def __init__(self, llm_model: str = "phi3-mini", embedding_model: str = "all-minilm-l6"):
        """Initialize the analyzer with selected models."""
        print("🤖 Initializing Codebase AI Analyzer...")
        
        # Validate model selection
        if llm_model not in AVAILABLE_MODELS:
            print(f"❌ Unknown LLM model: {llm_model}")
            print("Available models:", list(AVAILABLE_MODELS.keys()))
            raise ValueError(f"Invalid LLM model: {llm_model}")
            
        if embedding_model not in EMBEDDING_MODELS:
            print(f"❌ Unknown embedding model: {embedding_model}")
            print("Available models:", list(EMBEDDING_MODELS.keys()))
            raise ValueError(f"Invalid embedding model: {embedding_model}")
        
        self.llm_config = AVAILABLE_MODELS[llm_model]
        self.embedding_config = EMBEDDING_MODELS[embedding_model]
        
        print(f"📊 Selected LLM: {self.llm_config['description']}")
        print(f"📊 Selected Embedding: {self.embedding_config['description']}")
        
        # Initialize embedding model
        print("📥 Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.embedding_config['name'])
        
        # Initialize local vector database
        print("💾 Setting up vector database...")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=f"codebase_chunks_{embedding_model}",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize local LLM
        print(f"🧠 Loading language model: {self.llm_config['name']}...")
        print(f"⚠️  This will download ~{self.llm_config['size']} on first run")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Check if model requires authentication
            if self.llm_config.get('requires_auth', False):
                print(f"⚠️  {self.llm_config['name']} requires Hugging Face authentication")
                print("🔄 Falling back to DialoGPT...")
                raise Exception("Model requires authentication")
            
            self.llm = pipeline(
                "text-generation",
                model=self.llm_config['name'],
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"❌ Failed to load {self.llm_config['name']}: {e}")
            print("🔄 Falling back to DialoGPT...")
            self.llm = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        
        self.supported_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.cs', '.rb', '.go', '.rs', '.php', '.swift', '.kt',
            '.scala', '.clj', '.hs', '.ml', '.r', '.sql', '.sh',
            '.yaml', '.yml', '.json', '.xml', '.html', '.css',
            '.md', '.txt', '.dockerfile', '.makefile'
        }
        
        print("✅ Initialization complete!")
    
    def _get_file_hash(self, filepath: Path) -> str:
        """Generate hash for file content to track changes."""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _extract_code_chunks(self, filepath: Path, content: str) -> List[Dict[str, Any]]:
        """Extract meaningful chunks from code files."""
        chunks = []
        lines = content.split('\n')
        
        # Simple chunking strategy: group by functions/classes or fixed-size chunks
        current_chunk = []
        chunk_size = 50  # lines per chunk
        
        for i, line in enumerate(lines, 1):
            current_chunk.append(line)
            
            # Split on function/class definitions or when chunk gets too large
            if (len(current_chunk) >= chunk_size or 
                line.strip().startswith(('def ', 'class ', 'function ', 'const ', 'let ', 'var '))):
                
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append({
                        'content': chunk_content,
                        'filepath': str(filepath),
                        'start_line': i - len(current_chunk) + 1,
                        'end_line': i,
                        'file_type': filepath.suffix
                    })
                    current_chunk = []
        
        # Add remaining content
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'filepath': str(filepath),
                'start_line': len(lines) - len(current_chunk) + 1,
                'end_line': len(lines),
                'file_type': filepath.suffix
            })
        
        return chunks
    
    def index_codebase(self, codebase_path: str):
        """Index the entire codebase for search and retrieval."""
        codebase_path = Path(codebase_path)
        
        if not codebase_path.exists():
            print(f"❌ Error: Path {codebase_path} does not exist")
            return
        
        print(f"📂 Indexing codebase: {codebase_path}")
        
        # Clear existing collection
        try:
            # Get all existing documents and delete them
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
        except Exception as e:
            print(f"⚠️  Warning: Could not clear existing collection: {e}")
            # Try to recreate the collection
            try:
                self.chroma_client.delete_collection(name=self.collection.name)
                self.collection = self.chroma_client.create_collection(
                    name=f"codebase_chunks_{embedding_model}",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e2:
                print(f"⚠️  Warning: Could not recreate collection: {e2}")
        
        all_chunks = []
        file_count = 0
        
        # Walk through all files
        for root, dirs, files in os.walk(codebase_path):
            # Skip common ignore directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}]
            
            for file in files:
                filepath = Path(root) / file
                
                if filepath.suffix.lower() in self.supported_extensions:
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Extract chunks from this file
                        chunks = self._extract_code_chunks(filepath, content)
                        all_chunks.extend(chunks)
                        file_count += 1
                        
                        if file_count % 10 == 0:
                            print(f"📄 Processed {file_count} files...")
                    
                    except Exception as e:
                        print(f"⚠️  Warning: Could not read {filepath}: {e}")
        
        print(f"🔄 Creating embeddings for {len(all_chunks)} code chunks...")
        
        # Create embeddings and store in vector database
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            # Create embeddings
            texts = [chunk['content'] for chunk in batch]
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Prepare metadata
            metadatas = []
            ids = []
            
            for j, chunk in enumerate(batch):
                chunk_id = f"{chunk['filepath']}:{chunk['start_line']}-{chunk['end_line']}"
                ids.append(chunk_id)
                metadatas.append({
                    'filepath': chunk['filepath'],
                    'start_line': chunk['start_line'],
                    'end_line': chunk['end_line'],
                    'file_type': chunk['file_type']
                })
            
            # Add to vector database
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        print(f"✅ Successfully indexed {file_count} files with {len(all_chunks)} code chunks")
    
    def search_code(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant code chunks based on query."""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Format results
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return search_results
    
    def answer_question(self, question: str) -> str:
        """Answer a question about the codebase using RAG."""
        print(f"🔍 Searching for relevant code...")
        
        # Retrieve relevant code chunks
        relevant_chunks = self.search_code(question, top_k=3)
        
        if not relevant_chunks:
            return "❌ No relevant code found for your question."
        
        # Prepare context from retrieved chunks
        context = "\n\n---\n\n".join([
            f"File: {chunk['metadata']['filepath']} (lines {chunk['metadata']['start_line']}-{chunk['metadata']['end_line']})\n"
            f"```{chunk['metadata']['file_type']}\n{chunk['content']}\n```"
            for chunk in relevant_chunks
        ])
        
        # Create prompt for the LLM
        prompt = f"""Based on the following code context, please answer the question:

Question: {question}

Code Context:
{context}

Answer:"""
        
        print("🧠 Generating answer...")
        
        # Generate response using local LLM
        try:
            response = self.llm(
                prompt,
                max_length=len(prompt.split()) + 200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "❌ Could not generate a meaningful answer."
        
        except Exception as e:
            return f"❌ Error generating answer: {e}"
    
    def interactive_mode(self):
        """Start interactive question-answering mode."""
        print("\n🎯 Interactive Mode Started!")
        print("Ask questions about your codebase. Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\n" + "="*50)
                answer = self.answer_question(question)
                print(f"🤖 Answer: {answer}")
                print("="*50 + "\n")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Offline Codebase AI Analyzer")
    parser.add_argument("command", choices=["index", "query", "interactive", "list-models"], 
                       help="Command to execute")
    parser.add_argument("--path", 
                       help="Path to codebase directory")
    parser.add_argument("--question", 
                       help="Question to ask (for query command)")
    parser.add_argument("--llm", 
                       default="phi3-mini",
                       help="LLM model to use (default: phi3-mini)")
    parser.add_argument("--embedding", 
                       default="all-minilm-l6",
                       help="Embedding model to use (default: all-minilm-l6)")
    
    args = parser.parse_args()
    
    if args.command == "list-models":
        print("🤖 Available Models:")
        print("=" * 60)
        
        print("\n📊 LLM Models:")
        print("-" * 30)
        for model_id, model_info in AVAILABLE_MODELS.items():
            print(f"  {model_id}:")
            print(f"    {model_info['description']}")
            print(f"    Size: {model_info['size']} | Quality: {model_info['quality']} | Speed: {model_info['speed']}")
            print()
        
        print("🔍 Embedding Models:")
        print("-" * 30)
        for model_id, model_info in EMBEDDING_MODELS.items():
            print(f"  {model_id}:")
            print(f"    {model_info['description']}")
            print(f"    Size: {model_info['size']} | Dimensions: {model_info['dimensions']}")
            print()
        
        print("💡 Recommendations:")
        print("-" * 30)
        print("  🏆 Best Quality: codellama-7b, deepseek-coder")
        print("  ⚡ Fastest: gemma-2b, phi3-mini")
        print("  🎯 Balanced: mistral-7b, llama3-8b")
        return
    
    if not args.path:
        print("❌ Error: --path required for index, query, and interactive commands")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = CodebaseAnalyzer(llm_model=args.llm, embedding_model=args.embedding)
    
    if args.command == "index":
        analyzer.index_codebase(args.path)
    
    elif args.command == "query":
        if not args.question:
            print("❌ Error: --question required for query command")
            sys.exit(1)
        
        answer = analyzer.answer_question(args.question)
        print(f"🤖 Answer: {answer}")
    
    elif args.command == "interactive":
        # First ensure codebase is indexed
        print("🔄 Ensuring codebase is indexed...")
        analyzer.index_codebase(args.path)
        analyzer.interactive_mode()

if __name__ == "__main__":
    main()