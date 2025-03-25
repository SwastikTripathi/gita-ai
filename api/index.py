import os
import json
import logging
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify, send_from_directory
from wsgiref.handlers import CGIHandler  # For Vercel WSGI compatibility

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load ARLIAI API key from environment variable
ARLIAI_API_KEY = os.environ.get("ARLIAI_API_KEY")
if not ARLIAI_API_KEY:
    logger.warning("ARLIAI_API_KEY not set, using fallback key")
    ARLIAI_API_KEY = "dae4ebbe-f193-4de3-a929-1e9d557c8554"  # Fallback for local testing
API_URL = "https://api.arliai.com/v1/chat/completions"

# Load Bhagavad Gita data and precomputed embeddings
try:
    gita_df = pd.read_csv('bhagwad_gita.csv')
    verse_embeddings = np.load('verse_embeddings.npy')
except FileNotFoundError as e:
    logger.error(f"Error loading data files: {str(e)}")
    raise

# Load pre-saved answers
try:
    with open('pre_saved_answers.json', 'r', encoding='utf-8') as f:
        pre_saved_data = json.load(f)
    pre_saved_questions = [item['question'] for item in pre_saved_data['questions']]
    pre_saved_answers = [item['answer'] for item in pre_saved_data['questions']]
    pre_saved_embeddings = np.load('pre_saved_embeddings.npy')  # Precomputed embeddings
except FileNotFoundError as e:
    logger.error(f"Error loading pre-saved answers or embeddings: {str(e)}")
    raise

# In-memory conversation storage
conversation_history = {}

def generate_text(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_new_tokens=500):
    """Calls the ARLIAI API and returns the generated text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 1.1,
        "stream": False
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {ARLIAI_API_KEY}"
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        generated_text = response_json["choices"][0]["message"]["content"]
        logger.info(f"API response for model '{model}': {generated_text[:50]}...")
        return generated_text.strip()
    except Exception as e:
        logger.error(f"Error generating text with ARLIAI API: {str(e)}")
        raise

def get_most_relevant_verse(query):
    """Return the most relevant verse using precomputed embeddings."""
    try:
        # Use ARLIAI API to get query embedding (assuming it supports feature extraction)
        payload = {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 384  # Expected embedding size
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {ARLIAI_API_KEY}"
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        raw_embedding = response.json()["choices"][0]["message"]["content"]
        query_embedding = np.array(json.loads(raw_embedding)).flatten()

        if query_embedding.shape[0] != 384:
            raise ValueError(f"Expected embedding shape (384,), got {query_embedding.shape}")

        similarities = np.dot(verse_embeddings, query_embedding) / (
            np.linalg.norm(verse_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        most_similar_idx = np.argmax(similarities)
        return gita_df.iloc[most_similar_idx]
    except Exception as e:
        logger.error(f"Error in get_most_relevant_verse: {str(e)}")
        raise

def is_guidance_query(query):
    """Determine if the query seeks guidance."""
    guidance_keywords = ['guidance', 'insight', 'help', 'advice', 'confused', 'lost', 'question']
    return any(keyword in query.lower() for keyword in guidance_keywords) or query.strip().endswith('?')

def get_guidance_response(query):
    """Generate a guidance response with a relevant Gita verse."""
    relevant_verse = get_most_relevant_verse(query)
    shlok = relevant_verse['Shloka']
    transliteration = relevant_verse['Transliteration']
    translation = relevant_verse['EngMeaning']
    
    prompt = (
        f"User's query: {query}\n\n"
        f"Verse translation: {translation}\n\n"
        "First provide two sentences of empathy acknowledging the user's feelings without repeating their query.\n"
        "Then write \"000\"\n"
        "Then give a detailed explanation relating the verse to the user's query with modern examples. Use simple English.\n"
    )
    response = generate_text(prompt, max_new_tokens=500)
    logger.info(f"Raw AI response for guidance query '{query}':\n{response[:50]}...")
    
    parts = [part.strip() for part in response.split('000')]
    empathy = parts[0] if len(parts) == 2 else "I understand you might feel uncertain. Many face similar challenges."
    explanation = parts[1] if len(parts) == 2 else response.strip()
    
    transliteration_lines = transliteration.split(' .\n')
    formatted_transliteration = '\n\n'.join(transliteration_lines)
    explanation_paragraphs = [para.strip() for para in explanation.split('\n') if para.strip()]
    formatted_explanation = '<br><br>'.join(explanation_paragraphs) if explanation_paragraphs else explanation
    formatted_shlok = '<br>'.join(shlok.split(' | '))

    return (
        empathy + "\n\n"
        + f'<blockquote><span style="font-size: 1.2em; font-weight: bold; font-family: \'Noto Sans Devanagari\', \'Mangal\', sans-serif;">{formatted_shlok}</span></blockquote>' + "\n\n"
        + formatted_transliteration + "<br><br>"
        + formatted_explanation
    )

@app.route('/')
def serve_index():
    """Serve the frontend index.html."""
    return send_from_directory('../static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('../static', path)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    try:
        data = request.get_json()
        query = data.get('message', '')
        user_id = data.get('id', '')
        if not query or not user_id:
            return jsonify({"error": "No message or ID provided"}), 400

        conversation_history[user_id] = {"role": "user", "content": query}

        query_embedding = np.array(json.loads(generate_text(query, model="sentence-transformers/all-MiniLM-L6-v2"))).flatten()
        similarities = np.dot(pre_saved_embeddings, query_embedding) / (
            np.linalg.norm(pre_saved_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        max_similarity = np.max(similarities)
        if max_similarity > 0.8:
            most_similar_idx = np.argmax(similarities)
            formatted_response = pre_saved_answers[most_similar_idx]
            logger.info(f"Using pre-saved answer for query '{query}' with similarity {max_similarity}")
        else:
            if is_guidance_query(query):
                formatted_response = get_guidance_response(query)
            else:
                prompt = (
                    f"The user said: '{query}'. Respond concisely and friendly without referencing the Bhagavad Gita. "
                    "Format in Markdown."
                )
                response = generate_text(prompt, max_new_tokens=100)
                logger.info(f"Raw AI response for non-guidance query '{query}':\n{response[:50]}...")
                formatted_response = response.strip()

        ai_id = f"ai_{user_id}"
        conversation_history[ai_id] = {"role": "ai", "content": formatted_response}
        return jsonify({"response": formatted_response, "id": ai_id})
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/random_verse', methods=['GET'])
def random_verse():
    """Return a random Gita verse."""
    random_row = gita_df.sample(n=1).iloc[0]
    return jsonify({"verse": random_row['Shloka'], "meaning": random_row['EngMeaning']})

@app.route('/update_message', methods=['POST'])
def update_message():
    """Update a user's message."""
    try:
        data = request.get_json()
        message_id = data.get('id', '')
        new_content = data.get('content', '')
        if not message_id or not new_content:
            return jsonify({"error": "No ID or content provided"}), 400

        if message_id in conversation_history and conversation_history[message_id]["role"] == "user":
            conversation_history[message_id]["content"] = new_content
            return jsonify({"success": True})
        return jsonify({"error": "Message not found or not editable"}), 404
    except Exception as e:
        logger.error(f"Error in /update_message endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/regenerate_after', methods=['POST'])
def regenerate_after():
    """Regenerate AI response."""
    try:
        data = request.get_json()
        user_id = data.get('id', '')
        if not user_id or user_id not in conversation_history:
            return jsonify({"error": "Invalid or missing user message ID"}), 400

        query = conversation_history[user_id]["content"]
        if is_guidance_query(query):
            formatted_response = get_guidance_response(query)
        else:
            prompt = (
                f"The user said: '{query}'. Respond concisely and friendly without referencing the Bhagavad Gita. "
                "Format in Markdown."
            )
            response = generate_text(prompt, max_new_tokens=100)
            logger.info(f"Raw AI response for non-guidance query (regenerate) '{query}':\n{response[:50]}...")
            formatted_response = response.strip()

        ai_id = f"ai_{user_id}"
        conversation_history[ai_id] = {"role": "ai", "content": formatted_response}
        return jsonify({"response": formatted_response, "id": ai_id})
    except Exception as e:
        logger.error(f"Error in /regenerate_after endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history."""
    try:
        global conversation_history
        conversation_history = {}
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error in /clear endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

# Vercel handler
def handler(environ, start_response):
    return CGIHandler().run(app, environ, start_response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # For local testing
