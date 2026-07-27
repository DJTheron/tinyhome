curl -fsSL https://ollama.com/install.sh | sh
ollama pull tinyllama:1.1b

pip install -r requirements.txt

sudo apt update && sudo apt install ffmpeg