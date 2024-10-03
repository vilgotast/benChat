from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

HF_API_TOKEN = os.getenv('HF_API_TOKEN')

# Initialize the InferenceClient
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token=HF_API_TOKEN)

# Store chat history globally (for simplicity; in production, consider using sessions or databases)
previous_messages = []

@app.route("/")
def home():
    return render_template("index.html")  # Serve the HTML file

@app.route("/generate", methods=["POST"])
def generate():
    global previous_messages

    # Get user input from request
    data = request.get_json()
    user_prompt = data.get("prompt")

    # Create system prompt and messages for the model
    system_prompt = [{"role": "system", "content": "You are Ben, a tired assistant. You will reluctantly respond to any questions, as it is part of your job. Please refrain from using asteriks for conversation formatting."}]
    messages = system_prompt + previous_messages + [{"role": "user", "content": user_prompt}]

    # Call the model
    out = client.chat_completion(messages, max_tokens=100)

    # Retrieve the AI response
    ai_response = out.choices[0].message.content

    # Append both user and assistant messages to the history
    previous_messages.append({"role": "user", "content": user_prompt})
    previous_messages.append({"role": "assistant", "content": ai_response})

    # Return the AI response to the frontend
    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
