import os
import glob

def migrate_to_groq_model():
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_pattern = os.path.join(backend_dir, "**", "*.py")
    files = glob.glob(search_pattern, recursive=True)
    
    for file_path in files:
        if "venv" in file_path or ".venv" in file_path or "test_" in file_path:
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if "openai/gpt-oss-120b" in content:
            content = content.replace("openai/gpt-oss-120b", "openai/gpt-oss-120b")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated {file_path}")

if __name__ == "__main__":
    migrate_to_groq_model()
