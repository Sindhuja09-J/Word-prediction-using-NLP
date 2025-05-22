from flask import Flask, render_template, request, session
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Secure secret key

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Load sentiment analysis
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Detect incomplete sentences
def is_incomplete(sentence):
    return not re.search(r'[.!?]$', sentence.strip()) and len(sentence.strip().split()) > 2

# Suggest next word
def generate_next_word(input_sentence):
    inputs = tokenizer(input_sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        next_word = tokenizer.decode([next_token_id]).strip()
    return next_word

# Check for irrelevant input
def is_irrelevant_input(input_text):
    irrelevant_keywords = ["blah", "test", "xyz", "asdf", "random", "123", "dummy","location","nlp","recipe"]
    if any(keyword in input_text.lower() for keyword in irrelevant_keywords):
        return True
    if len(input_text.strip().split()) <= 1:  # Very short input
        return True
    return False

# Adjust response based on sentiment
def adjust_response_based_on_sentiment(response, sentiment):
    if sentiment == 'NEGATIVE':
        return "I'm really sorry you're feeling that way. Would you like to talk about it?"
    elif sentiment == 'POSITIVE':
        return f"That's great! 😊 {response}"
    else:
        return response

@app.route("/", methods=["GET", "POST"])
def chatbot():
    if "chat_history" not in session:
        session["chat_history"] = []
        session["chat_history_ids"] = None

    suggestion = ""
    response = ""

    if request.method == "POST":
        user_input = request.form["user_input"].strip()

        if not user_input:
            return render_template("index.html", active_tab="chatbot", conversation=session["chat_history"], suggestion="", response="")

        original_input = user_input

        if is_irrelevant_input(user_input):
            response = "I'm not sure how to respond to that. Could you please rephrase or ask something else?"
            session["chat_history"].append(("You", original_input))
            session["chat_history"].append(("Bot", response))
            return render_template("index.html", active_tab="chatbot", conversation=session["chat_history"], suggestion="", response=response)

        # Check if incomplete
        if is_incomplete(user_input):
            next_word = generate_next_word(user_input)
            suggestion = next_word
            user_input += " " + next_word

        # Append user input
        session["chat_history"].append(("You", original_input))

        # Prepare model input
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        if session["chat_history_ids"] is not None:
            previous_ids = torch.tensor(session["chat_history_ids"]).to(new_input_ids.device)
            bot_input_ids = torch.cat([previous_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids


        # Generate bot response
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=bot_input_ids.shape[-1] + 50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.5,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        session["chat_history_ids"] = chat_history_ids.tolist()

        # Decode response
        generated_tokens = chat_history_ids[:, bot_input_ids.shape[-1]:][0]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Fallback if response too short or empty
        if not response or len(response.split()) < 3:
            response = "I'm not sure how to respond to that. Could you please rephrase or ask something else?"

        # Analyze sentiment
        sentiment_result = sentiment_analyzer(original_input)
        sentiment = sentiment_result[0]['label']

        # Adjust based on sentiment
        response = adjust_response_based_on_sentiment(response, sentiment)

        # Append bot response
        session["chat_history"].append(("Bot", response))

    return render_template("index.html",
                           active_tab="chatbot",
                           conversation=session["chat_history"],
                           suggestion=suggestion,
                           response=response)

@app.route("/history")
def history():
    return render_template("index.html", active_tab="history", conversation=session.get("chat_history", []))

@app.route("/about")
def about():
    return render_template("index.html", active_tab="about")

if __name__ == "__main__":
    app.run(debug=True)
