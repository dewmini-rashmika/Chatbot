import os
import glob

def migrate_to_groq():
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_pattern = os.path.join(backend_dir, "**", "*.py")
    files = glob.glob(search_pattern, recursive=True)
    
    for file_path in files:
        if "venv" in file_path or ".venv" in file_path or "test_" in file_path:
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if "ChatGoogleGenerativeAI" in content:
            # Replace import
            content = content.replace("from langchain_groq import ChatGroq", "from langchain_groq import ChatGroq")
            
            # Replace instantiation
            content = content.replace("ChatGroq(", "ChatGroq(")
            content = content.replace("model="qwen/qwen3.8-27b",", "model=\"openai/gpt-oss-120b\",")
            content = content.replace("model="qwen/qwen3.8-27b",", "model=\"openai/gpt-oss-120b\",")
            content = content.replace("groq_api_key=settings.groq_api_key", "groq_api_key=settings.groq_api_key")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated {file_path}")

if __name__ == "__main__":
    migrate_to_groq()
