from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os
import logging
import json

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
    raise ValueError("ARLIAI_API_KEY environment variable is not set")
API_URL = "https://api.arliai.com/v1/chat/completions"

# Load data with absolute paths
BASE_DIR = os.path.dirname(__file__)
gita_df = pd.read_csv(os.path.join(BASE_DIR, 'bhagwad_gita.csv'))
verse_embeddings = np.load(os.path.join(BASE_DIR, 'verse_embeddings.npy'))
with open(os.path.join(BASE_DIR, 'pre_saved_answers.json'), 'r', encoding='utf-8') as f:
    pre_saved_data = json.load(f)
pre_saved_questions = [item['question'] for item in pre_saved_data['questions']]
pre_saved_answers = [item['answer'] for item in pre_saved_data['questions']]
pre_saved_embeddings = np.load(os.path.join(BASE_DIR, 'pre_saved_embeddings.npy'))

# Load SentenceTransformer model (only for queries)
try:
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {str(e)}")
    st_model = None

# In-memory conversation storage
conversation_history = {}

def generate_text(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_new_tokens=500):
    """Calls the Arliai API and returns the generated text."""
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
        response_json = response.json()
        generated_text = response_json["choices"][0]["message"]["content"]
        logger.info(f"API response for model '{model}': {generated_text}")
        return generated_text.strip()
    except Exception as e:
        logger.error(f"Error generating text with Arliai API: {str(e)}")
        raise

def get_most_relevant_verse(query):
    """Return the most relevant verse row based on precomputed embeddings."""
    if st_model is None:
        logger.warning(f"Fallback to random verse for query '{query}' due to SentenceTransformer failure")
        return gita_df.sample(n=1).iloc[0]
    
    try:
        query_embedding = st_model.encode([query], convert_to_tensor=False)[0]
        similarities = np.dot(verse_embeddings, query_embedding) / (
            np.linalg.norm(verse_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        most_similar_idx = np.argmax(similarities)
        return gita_df.iloc[most_similar_idx]
    except Exception as e:
        logger.error(f"Error encoding query '{query}' with SentenceTransformer: {str(e)}")
        return gita_df.sample(n=1).iloc[0]

def is_guidance_query(query):
    """Determine if the user's query is seeking guidance, advice, or wisdom."""
    guidance_keywords = ['guidance', 'insight', 'help', 'advice', 'confused', 'lost', 'question']
    return any(keyword in query.lower() for keyword in guidance_keywords) or query.strip().endswith('?')

def get_guidance_response(query):
    relevant_verse = get_most_relevant_verse(query)
    shlok = relevant_verse['Shloka']
    transliteration = relevant_verse['Transliteration']
    translation = relevant_verse['EngMeaning']
    
    prompt = (
        f"User's query: {query}\n\n"
        f"Verse translation: {translation}\n\n"
        "First provide two sentences of empathy acknowledging the user's feelings without repeating their query.\n"
        "Then write \"000\"\n"
        "Then give a detailed explanation that relates the verse to the user's query, including modern-day examples. Use simple English.\n"
    )
    
    response = generate_text(prompt, max_new_tokens=500)
    logger.info(f"Raw AI response for guidance query '{query}':\n{response}")
    
    parts = [part.strip() for part in response.split('000')]
    empathy = parts[0] if len(parts) == 2 else "I understand that you might be feeling uncertain or seeking guidance. Many people experience similar challenges in their lives."
    explanation = parts[1] if len(parts) == 2 else response.strip()
    
    formatted_shlok = '<br>'.join(shlok.split(' | '))
    formatted_transliteration = '\n\n'.join(transliteration.split(' .\n'))
    formatted_explanation = '<br><br>'.join([para.strip() for para in explanation.split('\n') if para.strip()] or [explanation])
    
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
        if not query or not user_id:
            return jsonify({"error": "No message or ID provided"}), 400

        conversation_history[user_id] = {"role": "user", "content": query}

        if st_model is not None:
            try:
                query_embedding = st_model.encode([query], convert_to_tensor=False)[0]
                similarities = np.dot(pre_saved_embeddings, query_embedding) / (
                    np.linalg.norm(pre_saved_embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
                max_similarity = np.max(similarities)
                if max_similarity > 0.8:
                    most_similar_idx = np.argmax(similarities)
                    formatted_response = pre_saved_answers[most_similar_idx]
                    logger.info(f"Using pre-saved answer for query '{query}' with similarity {max_similarity}")
                else:
                    formatted_response = None
            except Exception as e:
                logger.error(f"Error encoding query '{query}' with SentenceTransformer: {str(e)}")
                formatted_response = None
        else:
            logger.warning(f"Skipping pre-saved answer check due to SentenceTransformer failure")
            formatted_response = None

        if formatted_response is None:
            if is_guidance_query(query):
                formatted_response = get_guidance_response(query)
            else:
                prompt = (
                    f"The user said: '{query}'. Respond in a concise, friendly manner without referencing the Bhagavad Gita. "
                    "Format your response in Markdown for best readability."
                )
                formatted_response = generate_text(prompt, max_new_tokens=100).strip()

        ai_id = f"ai_{user_id}"
        conversation_history[ai_id] = {"role": "ai", "content": formatted_response}
        return jsonify({"response": formatted_response, "id": ai_id})
    
    except Exception as e:
        logger.error(f"Error in /api/chat endpoint: {str(e)}")
        fallback_response = "I’m having trouble generating a response right now. Please try again later."
        ai_id = f"ai_{user_id if 'user_id' in locals() else 'unknown'}"
        if 'user_id' in locals():
            conversation_history[ai_id] = {"role": "ai", "content": fallback_response}
        return jsonify({"response": fallback_response, "id": ai_id}), 500

@app.route('/api/random_verse', methods=['GET'])
def random_verse():
    random_row = gita_df.sample(n=1).iloc[0]
    return jsonify({"verse": random_row['Shloka'], "meaning": random_row['EngMeaning']})

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
        return jsonify({"error": "Message not found or not editable"}), 404

    except Exception as e:
        logger.error(f"Error in /api/update_message endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/api/regenerate_after', methods=['POST'])
def regenerate_after():
    try:
        data = request.json
        user_id = data.get('id', '')
        if not user_id or user_id not in conversation_history:
            return jsonify({"error": "Invalid or missing user message ID"}), 400

        query = conversation_history[user_id]["content"]
        if is_guidance_query(query):
            formatted_response = get_guidance_response(query)
        else:
            prompt = (
                f"The user said: '{query}'. Respond in a concise, friendly manner without referencing the Bhagavad Gita. "
                "Format your response in Markdown for best readability."
            )
            formatted_response = generate_text(prompt, max_new_tokens=100).strip()

        ai_id = f"ai_{user_id}"
        conversation_history[ai_id] = {"role": "ai", "content": formatted_response}
        return jsonify({"response": formatted_response, "id": ai_id})

    except Exception as e:
        logger.error(f"Error in /api/regenerate_after endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    try:
        global conversation_history
        conversation_history = {}
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error in /api/clear endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
