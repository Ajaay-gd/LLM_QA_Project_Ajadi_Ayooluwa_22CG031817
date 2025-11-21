"""
LLM Q&A Web GUI Application using Flask
With proper configuration and production-ready setup
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import re
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['JSON_SORT_KEYS'] = False

# Enable CORS for API endpoints
CORS(app)


class LLMQuestionAnswering:
    def __init__(self, api_key: str = None):
        """Initialize the LLM Q&A system"""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
    def preprocess_question(self, question: str) -> str:
        """Preprocess the input question"""
        processed = question.lower()
        processed = re.sub(r'\s+', ' ', processed)
        processed = re.sub(r'[^a-z0-9\s\?\.\,\!\-]', '', processed)
        processed = processed.strip()
        return processed
    
    def tokenize(self, text: str) -> list:
        """Simple word tokenization"""
        return text.split()
    
    def create_prompt(self, question: str) -> str:
        """Create a structured prompt for the LLM"""
        prompt = f"""You are a helpful AI assistant. Answer the following question clearly and concisely.

Question: {question}

Answer:"""
        return prompt
    
    def query_llm(self, question: str) -> dict:
        """Send question to LLM API and get response"""
        if not self.api_key:
            return {
                "error": "API key not configured. Please set GROQ_API_KEY environment variable.",
                "success": False
            }
        
        processed_question = self.preprocess_question(question)
        prompt = self.create_prompt(question)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            answer = data["choices"][0]["message"]["content"]
            
            return {
                "original_question": question,
                "processed_question": processed_question,
                "tokens": self.tokenize(processed_question),
                "token_count": len(self.tokenize(processed_question)),
                "answer": answer,
                "success": True
            }
        
        except requests.exceptions.Timeout:
            return {
                "error": "Request timed out. Please try again.",
                "success": False
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": f"API request failed: {str(e)}",
                "success": False
            }
        except (KeyError, IndexError) as e:
            return {
                "error": f"Failed to parse API response: {str(e)}",
                "success": False
            }


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "LLM Q&A Application"
    })


@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question submission"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided."
            }), 400
        
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                "success": False,
                "error": "Please enter a valid question."
            }), 400
        
        # Initialize Q&A system
        qa_system = LLMQuestionAnswering()
        
        # Get response
        result = qa_system.query_llm(question)
        
        if result.get('success'):
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error"
    }), 500


if __name__ == '__main__':
    # Get configuration from environment variables
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    print("="*60)
    print("🤖 LLM Q&A Flask Application")
    print("="*60)
    print(f"Server: http://{host}:{port}")
    print(f"Debug Mode: {debug}")
    print(f"API Key Configured: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")
    print("="*60)
    
    app.run(host=host, port=port, debug=debug)
