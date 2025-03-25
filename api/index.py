from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os
import logging
import json
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load ARLIAI API key from environment variable (set in Vercel dashboard)
ARLIAI_API_KEY = os.environ.get("ARLIAI_API_KEY")
if not ARLIAI_API_KEY:
    raise ValueError("ARLIAI_API_KEY environment variable is not set")
API_URL = "https://api.arliai.com/v1/chat/completions"

# Load Bhagavad Gita data and embeddings using absolute paths
gita_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'bhagwad_gita.csv'))
verse_embeddings = np.load(os.path.join(os.path.dirname(__file__), 'verse_embeddings.npy'))
with open(os.path.join(os.path.dirname(__file__), 'pre_saved_answers.json'), 'r', encoding='utf-8') as f:
    pre_saved_data = json.load(f)
pre_saved_questions = [item['question'] for item in pre_saved_data['questions']]
pre_saved_answers = [item['answer'] for item in pre_saved_data['questions']]
pre_saved_embeddings = np.load(os.path.join(os.path.dirname(__file__), 'pre_saved_embeddings.npy'))

# Initialize SentenceTransformer for queries
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# File to store user queries (use /tmp for Vercel's writable filesystem)
QUERY_LOG_FILE = '/tmp/user_queries.json'

def log_user_query(query, user_id):
    """Append user query to a JSON file in /tmp."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "query": query
    }
    try:
        if os.path.exists(QUERY_LOG_FILE):
            with open(QUERY_LOG_FILE, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=2)
        else:
            with open(QUERY_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump([entry], f, indent=2)
    except Exception as e:
        logger.error(f"Error logging query: {str(e)}")
        # Fallback: append as a new line in case of issues
        with open(QUERY_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')

def generate_text(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_new_tokens=500):
    """Call Arliai API for text generation."""
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
        response = requests.post(API_URL, headers=headers, json=payload, timeout=8)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error with Arliai API: {str(e)}")
        raise

def get_most_relevant_verse(query):
    """Find the most relevant Gita verse."""
    query_embedding = st_model.encode([query], convert_to_tensor=False)[0]
    similarities = np.dot(verse_embeddings, query_embedding) / (
        np.linalg.norm(verse_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    most_similar_idx = np.argmax(similarities)
    return gita_df.iloc[most_similar_idx]

def is_guidance_query(query):
    """Check if query seeks guidance."""
    guidance_keywords = ['guidance', 'insight', 'help', 'advice', 'confused', 'lost', 'question']
    return any(keyword in query.lower() for keyword in guidance_keywords) or query.strip().endswith('?')

def get_guidance_response(query):
    """Generate guidance response with verse."""
    relevant_verse = get_most_relevant_verse(query)
    shlok = relevant_verse['Shloka']
    transliteration = relevant_verse['Transliteration']
    translation = relevant_verse['EngMeaning']
    
    prompt = (
        f"User's query: {query}\n\n"
        f"Verse translation: {translation}\n\n"
        "First provide two sentences of empathy acknowledging the user's feelings without repeating their query.\n"
        "Then write \"000\"\n"
        "Then give a detailed explanation relating the verse to the user's query with modern examples. Use simple English."
    )
    
    response = generate_text(prompt, max_new_tokens=500)
    parts = [part.strip() for part in response.split('000')]
    empathy = parts[0] if len(parts) == 2 else "It’s totally normal to feel unsure sometimes. Lots of us face similar moments."
    explanation = parts[1] if len(parts) == 2 else response.strip()
    
    formatted_shlok = '<br>'.join(shlok.split(' | '))
    formatted_transliteration = '\n\n'.join(transliteration.split(' .\n'))
    formatted_explanation = '<br><br>'.join([p.strip() for p in explanation.split('\n') if p.strip()])
    
    return (
        f"{empathy}\n\n"
        f'<blockquote><span style="font-size: 1.2em; font-weight: bold; font-family: \'Noto Sans Devanagari\', \'Mangal\', sans-serif;">{formatted_shlok}</span></blockquote>\n\n'
        f"{formatted_transliteration}<br><br>{formatted_explanation}"
    )

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('message', '')
        user_id = data.get('id', '')
        history = data.get('history', [])  # Client sends history (optional)

        if not query or not user_id:
            return jsonify({"error": "No message or ID provided"}), 400

        # Log the query
        log_user_query(query, user_id)

        # Check pre-saved answers
        query_embedding = st_model.encode([query], convert_to_tensor=False)[0]
        similarities = np.dot(pre_saved_embeddings, query_embedding) / (
            np.linalg.norm(pre_saved_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        max_similarity = np.max(similarities)
        if max_similarity > 0.8:
            most_similar_idx = np.argmax(similarities)
            formatted_response = pre_saved_answers[most_similar_idx]
            logger.info(f"Pre-saved answer used for '{query}' (similarity: {max_similarity})")
        else:
            if is_guidance_query(query):
                formatted_response = get_guidance_response(query)
            else:
                prompt = (
                    f"The user said: '{query}'. Respond concisely and friendly, no Bhagavad Gita references. "
                    f"Format in Markdown."
                )
                formatted_response = generate_text(prompt, max_new_tokens=100)

        ai_id = f"ai_{user_id}"
        return jsonify({"response": formatted_response, "id": ai_id})
    except Exception as e:
        logger.error(f"Error in /api/chat: {str(e)}")
        return jsonify({"response": "Sorry, something went wrong!", "id": f"ai_{user_id if 'user_id' in locals() else 'unknown'}"}), 500

@app.route('/api/random_verse', methods=['GET'])
def random_verse():
    random_row = gita_df.sample(n=1).iloc[0]
    return jsonify({"verse": random_row['Shloka'], "meaning": random_row['EngMeaning']})

@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    return jsonify({"success": True})  # Client handles clearing

if __name__ == '__main__':
    app.run(debug=True, port=5000)
