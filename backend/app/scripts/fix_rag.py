import os

def update_rag_agent():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rag_file = os.path.join(base_dir, "app", "agents", "rag_agent.py")
    
    with open(rag_file, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Lower vector results
    if "n_results=10" in content:
        content = content.replace("n_results=10", "n_results=3")
        
    # Truncate context string to 3000 chars roughly to save tokens
    target = "context_str = \"\\n\\n---\\n\\n\".join(\n        [f\"[Source: {doc.get('source', 'Unknown')}]\\n{doc.get('content', '')}\"\n         for doc in fused_context]\n    )"
    replacement = target + "\n\n    # Truncate context to fit in token limits\n    if len(context_str) > 10000:\n        context_str = context_str[:10000] + '... [TRUNCATED]'"
    
    if "len(context_str) > 10000" not in content:
        content = content.replace(target, replacement)
        
    with open(rag_file, "w", encoding="utf-8") as f:
        f.write(content)
    print("Updated rag_agent.py")

if __name__ == "__main__":
    update_rag_agent()
