import glob
files = glob.glob('app/agents/*.py')
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    new_content = content.replace('llama3-8b-8192', 'qwen/qwen3.8-27b')
    new_content = new_content.replace('model=\'qwen/qwen3.8-27b\',', 'model=\'qwen/qwen3.8-27b\',\n    max_tokens=900,')
    new_content = new_content.replace('model=\"qwen/qwen3.8-27b\",', 'model=\"qwen/qwen3.8-27b\",\n    max_tokens=900,')
    if new_content != content:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print('Updated ' + file)
