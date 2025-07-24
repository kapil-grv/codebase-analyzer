#!/usr/bin/env python3
"""
Web UI for Codebase AI Analyzer
A simple Flask web interface for querying codebases.
"""

from flask import Flask, render_template, request, jsonify, session
import os
import tempfile
import shutil
from pathlib import Path
from analyzer import CodebaseAnalyzer
import threading
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Global variables to store analyzer instances
analyzers = {}
analyzer_locks = {}

@app.route('/')
def index():
    """Main page with codebase path input and query interface."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_codebase():
    """Initialize analysis for a codebase path."""
    data = request.get_json()
    codebase_path = data.get('codebase_path')
    llm_model = data.get('llm_model', 'phi3-mini')
    embedding_model = data.get('embedding_model', 'all-minilm-l6')
    
    if not codebase_path:
        return jsonify({'error': 'Codebase path is required'}), 400
    
    # Convert Windows path to container path if needed
    if codebase_path.startswith('C:\\') or codebase_path.startswith('C:/'):
        # Convert Windows path to container path
        # C:\Users\kapil\Documents\Git\sss-backend -> /mnt/c/Users/kapil/Documents/Git/sss-backend
        container_path = codebase_path.replace('C:\\', '/mnt/c/').replace('C:/', '/mnt/c/').replace('\\', '/')
    else:
        container_path = codebase_path
    
    # Validate path exists
    if not os.path.exists(container_path):
        return jsonify({'error': f'Path does not exist: {container_path}'}), 400
    
    # Create a unique session ID for this analysis
    session_id = f"analysis_{int(time.time())}"
    session['session_id'] = session_id
    
    try:
        # Initialize analyzer in a separate thread
        def init_analyzer():
            try:
                analyzer = CodebaseAnalyzer(llm_model=llm_model, embedding_model=embedding_model)
                analyzer.index_codebase(container_path)
                analyzers[session_id] = analyzer
                analyzer_locks[session_id] = threading.Lock()
            except Exception as e:
                print(f"Error initializing analyzer: {e}")
        
        thread = threading.Thread(target=init_analyzer)
        thread.start()
        
        return jsonify({
            'session_id': session_id,
            'status': 'initializing',
            'message': 'Analyzer is being initialized. This may take a few minutes for large codebases.'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to initialize analyzer: {str(e)}'}), 500

@app.route('/api/query', methods=['POST'])
def query_codebase():
    """Query the analyzed codebase."""
    data = request.get_json()
    question = data.get('question')
    session_id = session.get('session_id')
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    if not session_id or session_id not in analyzers:
        return jsonify({'error': 'No active analysis session. Please analyze a codebase first.'}), 400
    
    try:
        with analyzer_locks[session_id]:
            analyzer = analyzers[session_id]
            answer = analyzer.answer_question(question)
            
        return jsonify({
            'answer': answer,
            'question': question
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to process query: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    """Get the status of the current analysis session."""
    session_id = session.get('session_id')
    
    if not session_id:
        return jsonify({'status': 'no_session'})
    
    if session_id in analyzers:
        return jsonify({'status': 'ready'})
    else:
        return jsonify({'status': 'initializing'})

@app.route('/api/models')
def get_models():
    """Get available models."""
    from analyzer import AVAILABLE_MODELS, EMBEDDING_MODELS
    
    return jsonify({
        'llm_models': AVAILABLE_MODELS,
        'embedding_models': EMBEDDING_MODELS
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 