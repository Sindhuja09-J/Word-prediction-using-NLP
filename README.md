# Word-prediction-using-NLP
This project is a web-based intelligent chatbot that uses pre-trained language models to respond to user inputs. It enhances responses based on sentiment and offers next-word suggestions when the input seems incomplete.

## 🔍 Features

- Chatbot built using **DialoGPT** (Microsoft)
- **Sentiment analysis** using `distilbert-base-uncased-finetuned-sst-2-english`
- Next-word prediction for incomplete messages
- Filters irrelevant/random input
- Maintains **conversation history**
- Web UI powered by **Flask**

## ▶️ How to Run

1. **Install dependencies**:
pip install flask transformers torch
2.Run the app:
python app.py
3.Open http://127.0.0.1:5000 in your browser.
