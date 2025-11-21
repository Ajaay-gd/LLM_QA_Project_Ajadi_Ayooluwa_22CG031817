"""
LLM Q&A CLI Application
Accepts natural language questions, preprocesses them, and gets answers from an LLM API
"""

import os
import re
import requests
from typing import Optional

class LLMQuestionAnswering:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM Q&A system
        Uses Groq API (free tier available) - you can swap for OpenAI, Cohere, etc.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
    def preprocess_question(self, question: str) -> str:
        """
        Preprocess the input question:
        - Lowercase
        - Remove extra spaces
        - Remove special characters (keep basic punctuation)
        """
        # Convert to lowercase
        processed = question.lower()
        
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Remove special characters but keep basic punctuation
        processed = re.sub(r'[^a-z0-9\s\?\.\,\!\-]', '', processed)
        
        # Strip leading/trailing spaces
        processed = processed.strip()
        
        return processed
    
    def tokenize(self, text: str) -> list:
        """Simple word tokenization"""
        return text.split()
    
    def create_prompt(self, question: str) -> str:
        """
        Create a structured prompt for the LLM
        """
        prompt = f"""You are a helpful AI assistant. Answer the following question clearly and concisely.

Question: {question}

Answer:"""
        return prompt
    
    def query_llm(self, question: str) -> dict:
        """
        Send question to LLM API and get response
        """
        if not self.api_key:
            return {
                "error": "API key not found. Please set GROQ_API_KEY environment variable or pass it to constructor."
            }
        
        processed_question = self.preprocess_question(question)
        prompt = self.create_prompt(question)  # Use original for better context
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",  # Fast, free Groq model
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            answer = data["choices"][0]["message"]["content"]
            
            return {
                "original_question": question,
                "processed_question": processed_question,
                "tokens": self.tokenize(processed_question),
                "answer": answer,
                "success": True
            }
        
        except requests.exceptions.RequestException as e:
            return {
                "error": f"API request failed: {str(e)}",
                "success": False
            }
    
    def display_result(self, result: dict):
        """
        Display the results in a formatted way
        """
        print("\n" + "="*60)
        
        if not result.get("success", False):
            print("ERROR:", result.get("error", "Unknown error"))
            print("="*60 + "\n")
            return
        
        print("ORIGINAL QUESTION:")
        print(f"  {result['original_question']}")
        print("\nPROCESSED QUESTION:")
        print(f"  {result['processed_question']}")
        print("\nTOKENS:")
        print(f"  {result['tokens']}")
        print("\nANSWER:")
        print(f"  {result['answer']}")
        print("="*60 + "\n")


def main():
    """
    Main CLI application loop
    """
    print("="*60)
    print("        LLM Question Answering CLI Application")
    print("="*60)
    print("\nThis application uses Groq's free LLM API.")
    print("Make sure to set your GROQ_API_KEY environment variable.")
    print("\nGet your free API key at: https://console.groq.com/")
    print("\nType 'exit' or 'quit' to end the session.\n")
    
    # Initialize the Q&A system
    qa_system = LLMQuestionAnswering()
    
    while True:
        # Get user input
        question = input("Enter your question: ").strip()
        
        # Check for exit commands
        if question.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using LLM Q&A CLI. Goodbye!\n")
            break
        
        # Skip empty questions
        if not question:
            print("Please enter a valid question.\n")
            continue
        
        # Process and display result
        result = qa_system.query_llm(question)
        qa_system.display_result(result)


if __name__ == "__main__":
    main()
