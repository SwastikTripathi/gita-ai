from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
import os

# Initialize Flask app
app = Flask(__name__)

# Hugging Face API Key
HF_API_KEY = os.environ.get("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY environment variable is not set")
client = InferenceClient(token=HF_API_KEY)

# Load Gita data and precomputed embeddings
gita_df = pd.read_csv('api/bhagwad_gita.csv')
verse_embeddings = np.load('api/verse_embeddings.npy')

# In-memory conversation storage
conversation_history = {}

# Find the most relevant verse using Hugging Face Inference API for query embedding
def get_most_relevant_verse(query):
    # Use the Inference API to get the embedding for the query
    query_embedding = client.feature_extraction(
        text=query,  # Changed from 'inputs' to 'text'
        model="sentence-transformers/all-MiniLM-L6-v2"
    )[0]  # Get the first embedding vector
    similarities = np.dot(verse_embeddings, query_embedding) / (np.linalg.norm(verse_embeddings, axis=1) * np.linalg.norm(query_embedding))
    most_similar_idx = np.argmax(similarities)
    return gita_df.iloc[most_similar_idx]

# Serve the frontend
@app.route('/')
def serve_index():
    return send_from_directory('../static', 'index.html')

# Handle chat
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('message', '')
        user_id = data.get('id', '')
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
        print(f"Error in chat: {str(e)}")
        return jsonify({"error": "Something went wrong on the server"}), 500

# Update a message
@app.route('/api/update_message', methods=['POST'])
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
        print(f"Error in update_message: {str(e)}")
        return jsonify({"error": "Something went wrong on the server"}), 500

# Regenerate AI response after editing
@app.route('/api/regenerate_after', methods=['POST'])
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
        print(f"Error in regenerate_after: {str(e)}")
        return jsonify({"error": "Something went wrong on the server"}), 500

# Clear conversation
@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    try:
        global conversation_history
        conversation_history = {}
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error in clear_conversation: {str(e)}")
        return jsonify({"error": "Something went wrong on the server"}), 500
