#!/bin/bash

echo "🐳 Setting up Dockerized Codebase AI Analyzer..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first:"
    echo "   https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose found"

# Create example codebase directory if it doesn't exist
if [ ! -d "example_codebase" ]; then
    echo "📁 Creating example codebase directory..."
    mkdir -p example_codebase
    
    # Create a simple Python example
    cat > example_codebase/main.py << 'EOF'
"""
Example Python application for testing the AI analyzer.
"""

class UserManager:
    def __init__(self):
        self.users = {}
    
    def create_user(self, username, email):
        """Create a new user account."""
        if username in self.users:
            raise ValueError("User already exists")
        
        self.users[username] = {
            'email': email,
            'active': True,
            'created_at': datetime.now()
        }
        return True
    
    def authenticate_user(self, username, password):
        """Authenticate user login."""
        user = self.users.get(username)
        if not user or not user.get('active'):
            return False
        
        # In real app, check password hash
        return True

def main():
    """Main application entry point."""
    manager = UserManager()
    
    # Example usage
    manager.create_user("john_doe", "john@example.com")
    
    if manager.authenticate_user("john_doe", "password"):
        print("Login successful!")
    else:
        print("Login failed!")

if __name__ == "__main__":
    main()
EOF

    # Create a JavaScript example
    cat > example_codebase/api.js << 'EOF'
/**
 * Simple REST API example
 */

const express = require('express');
const app = express();

// Middleware
app.use(express.json());

// In-memory database simulation
let users = [];
let nextId = 1;

// Routes
app.get('/api/users', (req, res) => {
    res.json(users);
});

app.post('/api/users', (req, res) => {
    const { name, email } = req.body;
    
    if (!name || !email) {
        return res.status(400).json({ error: 'Name and email required' });
    }
    
    const user = {
        id: nextId++,
        name,
        email,
        createdAt: new Date()
    };
    
    users.push(user);
    res.status(201).json(user);
});

app.get('/api/users/:id', (req, res) => {
    const user = users.find(u => u.id === parseInt(req.params.id));
    
    if (!user) {
        return res.status(404).json({ error: 'User not found' });
    }
    
    res.json(user);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
EOF

    echo "✅ Created example codebase with Python and JavaScript files"
fi

# Create directories for persistence
mkdir -p chroma_db huggingface_cache

echo "🔨 Building Docker image..."
docker compose build

echo "✅ Setup complete!"
echo ""
echo "🤖 Available Models:"
docker compose run --rm codebase-analyzer python3 analyzer.py list-models
echo ""
echo "🎯 Usage Commands:"
echo ""
echo "1. 📋 List all available models:"
echo "   docker compose run --rm codebase-analyzer python3 analyzer.py list-models"
echo ""
echo "2. 📊 Index your codebase (choose your model):"
echo "   docker compose run --rm codebase-analyzer python3 analyzer.py index --path /app/codebase --llm phi3-mini"
echo "   docker compose run --rm codebase-analyzer python3 analyzer.py index --path /app/codebase --llm codellama-7b"
echo ""
echo "3. 🤖 Interactive mode with different models:"
echo "   # Fast and lightweight"
echo "   docker compose run --rm codebase-analyzer python3 analyzer.py interactive --path /app/codebase --llm gemma-2b"
echo ""
echo "   # Best for code analysis" 
echo "   docker compose run --rm codebase-analyzer python3 analyzer.py interactive --path /app/codebase --llm codellama-7b --embedding code-search"
echo ""
echo "   # Balanced performance"
echo "   docker compose run --rm codebase-analyzer python3 analyzer.py interactive --path /app/codebase --llm phi3-mini"
echo ""
echo "4. ❓ Single questions:"
echo "   docker compose run --rm codebase-analyzer python3 analyzer.py query --path /app/codebase --llm mistral-7b --question \"How does authentication work?\""
echo ""
echo "5. 🚀 Quick presets:"
echo "   docker compose up fast-analyzer    # Gemma 2B (fastest)"
echo "   docker compose up code-analyzer    # CodeLlama 7B (best for code)"
echo ""
echo "🛠️  To analyze your own codebase:"
echo "   - Edit docker compose.yml"
echo "   - Change the volume mount: /path/to/your/codebase:/app/codebase:ro"
echo "   - Or copy your code to ./example_codebase/"
echo ""
echo "💡 Model Recommendations:"
echo "   🏆 Best Quality: codellama-7b, deepseek-coder"
echo "   ⚡ Fastest: gemma-2b, phi3-mini"  
echo "   🎯 Balanced: mistral-7b, llama3-8b"
echo ""
echo "💡 Quick start with example code:"
echo "   docker compose run --rm codebase-analyzer python3 analyzer.py interactive --path /app/codebase"
echo ""