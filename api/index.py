import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from huggingface_hub import InferenceClient

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from Vercel frontend

# Hugging Face API Key from environment variable
HF_API_KEY = os.environ.get("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY environment variable not set")
client = InferenceClient(token=HF_API_KEY)

# Load Gita data (assumed to be in root directory)
gita_df = pd.read_csv('../bhagwad_gita.csv')

# Load Sentence Transformer model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute verse embeddings
verse_meanings = gita_df['EngMeaning'].tolist()
verse_embeddings = st_model.encode(verse_meanings, convert_to_tensor=False)
verse_embeddings = np.array(verse_embeddings)

# In-memory conversation storage
conversation_history = {}

# Find the most relevant verse
def get_most_relevant_verse(query):
    query_embedding = st_model.encode([query], convert_to_tensor=False)[0]
    similarities = np.dot(verse_embeddings, query_embedding) / (np.linalg.norm(verse_embeddings, axis=1) * np.linalg.norm(query_embedding))
    most_similar_idx = np.argmax(similarities)
    return gita_df.iloc[most_similar_idx]

# Handle chat
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('message', '')
        user_id = data.get('id', '')  # Unique ID from frontend
        if not query or not user_id:
            return jsonify({"error": "No message or ID provided"}), 400

        # Store user message in history
        conversation_history[user_id] = {"role": "user", "content": query}

        if query.endswith('?'):
            # Gita quote logic
            relevant_verse = get_most_relevant_verse(query)
            shloka = relevant_verse['Shloka']
            transliteration = relevant_verse['Transliteration']
            translation = relevant_verse['EngMeaning']
            
            prompt = (
                f"User query: {query}\n\n"
                f"Relevant verse from the Bhagavad Gita:\n"
                f"Shloka: {shloka}\n"
                f"Transliteration: {transliteration}\n"
                f"Translation: {translation}\n\n"
                f"Provide a brief explanation (2-3 sentences) of how this verse relates to the user's query and what lessons can be learned from it in this context. "
                f"Format your response in Markdown."
            )
            
            response = client.text_generation(
                prompt=prompt,
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            explanation = response.strip()
            formatted_response = (
                f"### Hindi Shlok\n{shloka}\n\n"
                f"### Transliteration\n{transliteration}\n\n"
                f"### Explanation\n{explanation}"
            )
        else:
            # Normal conversational response
            prompt = (
                f"The user said: '{query}'. "
                f"Respond in a concise, friendly manner without referencing the Bhagavad Gita. "
                f"Format your response in Markdown."
            )
            
            response = client.text_generation(
                prompt=prompt,
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            formatted_response = response.strip()

        # Store AI response in history with a new ID
        ai_id = f"ai_{user_id}"
        conversation_history[ai_id] = {"role": "ai", "content": formatted_response}

        return jsonify({"response": formatted_response, "id": ai_id})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

# Update a message
@app.route('/update_message', methods=['POST'])
def update_message():
    try:
        data = request.json
        message_id = data.get('id', '')
        new_content = data.get('content', '')

        if not message_id or not new_content:
            return jsonify({"error": "No ID or content provided"}), 400

        if message_id in conversation_history and conversation_history[message_id]["role"] == "user":
            conversation_history[message_id]["content"] = new_content
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Message not found or not editable"}), 404

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

# Regenerate AI response after editing
@app.route('/regenerate_after', methods=['POST'])
def regenerate_after():
    try:
        data = request.json
        user_id = data.get('id', '')

        if not user_id or user_id not in conversation_history:
            return jsonify({"error": "Invalid or missing user message ID"}), 400

        query = conversation_history[user_id]["content"]

        if query.endswith('?'):
            relevant_verse = get_most_relevant_verse(query)
            shloka = relevant_verse['Shloka']
            transliteration = relevant_verse['Transliteration']
            translation = relevant_verse['EngMeaning']
            
            prompt = (
                f"User query: {query}\n\n"
                f"Relevant verse from the Bhagavad Gita:\n"
                f"Shloka: {shloka}\n"
                f"Transliteration: {transliteration}\n"
                f"Translation: {translation}\n\n"
                f"Provide a brief explanation (2-3 sentences) of how this verse relates to the user's query and what lessons can be learned from it in this context. "
                f"Format your response in Markdown."
            )
            
            response = client.text_generation(
                prompt=prompt,
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            explanation = response.strip()
            formatted_response = (
                f"### Hindi Shlok\n{shloka}\n\n"
                f"### Transliteration\n{transliteration}\n\n"
                f"### Explanation\n{explanation}"
            )
        else:
            prompt = (
                f"The user said: '{query}'. "
                f"Respond in a concise, friendly manner without referencing the Bhagavad Gita. "
                f"Format your response in Markdown."
            )
            
            response = client.text_generation(
                prompt=prompt,
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            formatted_response = response.strip()

        # Update AI response in history
        ai_id = f"ai_{user_id}"
        conversation_history[ai_id] = {"role": "ai", "content": formatted_response}

        return jsonify({"response": formatted_response, "id": ai_id})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

# Clear conversation
@app.route('/clear', methods=['POST'])
def clear_conversation():
    try:
        global conversation_history
        conversation_history = {}
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT env var if set (common in hosting platforms)
    app.run(host='0.0.0.0', port=port, debug=False)  # Run on 0.0.0.0 for external access
