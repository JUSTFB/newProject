import json

notebook_path = r"d:\newProject\colab_dubbing.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Fix install cell
        if any('pip install' in line for line in source) and any('TTS' in line for line in source):
            cell['source'] = [
                "# Install system dependencies\n",
                "!sudo apt-get -y install espeak-ng libsndfile1-dev\n",
                "\n",
                "# Install the community-maintained TTS fork (supports Python 3.12)\n",
                "!pip install -q coqui-tts\n",
                "\n",
                "# Install other dependencies\n",
                "!pip install -q faster-whisper deep-translator google-generativeai ffmpeg-python edge-tts gtts demucs flask pyngrok\n",
                "\n",
                "print(\"\\n\u2705 All packages installed!\")"
            ]
            print("Fixed install cell")
        
        # Fix total_mem -> total_memory
        new_source = []
        for line in source:
            new_source.append(line.replace('.total_mem ', '.total_memory ').replace('.total_mem/', '.total_memory/'))
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4)

print("Done - colab_dubbing.ipynb patched!")
