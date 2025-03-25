from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
import os
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load ARLIAI API key securely from an environment variable; no fallback for security
ARLIAI_API_KEY = os.environ.get("ARLIAI_API_KEY")
if not ARLIAI_API_KEY:
    raise ValueError("ARLIAI_API_KEY environment variable is not set")
API_URL = "https://api.arliai.com/v1/chat/completions"

# Define the text generation function using ARLIAI API
def generate_text(prompt, model):
    """
    Calls the ARLIAI API directly and returns the generated text.
    Parameters are configured on the ARLIAI website, so only prompt and model are passed.
    """
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {ARLIAI_API_KEY}"
    }
    try:
        response = requests.post(API_URL, headers=headers, data=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_json = response.json()
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"Unexpected response format: {response_json}")
            raise ValueError("Unexpected response format from ARLIAI API")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling ARLIAI API: {str(e)}")
        raise

# Load Bhagavad Gita data from CSV
gita_df = pd.read_csv('bhagwad_gita.csv')

# Load Sentence Transformer model and compute embeddings for verse meanings
st_model = SentenceTransformer('all-MiniLM-L6-v2')
verse_meanings = gita_df['EngMeaning'].tolist()
verse_embeddings = st_model.encode(verse_meanings, convert_to_tensor=False)
verse_embeddings = np.array(verse_embeddings)

# Load pre-saved answers for common queries
with open('pre_saved_answers.json', 'r', encoding='utf-8') as f:
    pre_saved_data = json.load(f)
pre_saved_questions = [item['question'] for item in pre_saved_data['questions']]
pre_saved_answers = [item['answer'] for item in pre_saved_data['questions']]
pre_saved_embeddings = st_model.encode(pre_saved_questions, convert_to_tensor=False)
pre_saved_embeddings = np.array(pre_saved_embeddings)

# In-memory conversation storage (consider a persistent DB for production)
conversation_history = {}

# Helper functions
def get_most_relevant_verse(query):
    """Return the most relevant verse row from the Bhagavad Gita based on the query."""
    query_embedding = st_model.encode([query], convert_to_tensor=False)[0]
    similarities = np.dot(verse_embeddings, query_embedding) / (
        np.linalg.norm(verse_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    most_similar_idx = np.argmax(similarities)
    return gita_df.iloc[most_similar_idx]

def is_guidance_query(query):
    """Determine if the user's query is seeking guidance, advice, or wisdom."""
    guidance_keywords = ['guidance', 'insight', 'help', 'advice', 'confused', 'lost', 'question']
    return any(keyword in query.lower() for keyword in guidance_keywords) or query.strip().endswith('?')

def get_guidance_response(query):
    """Generate a guidance response using a relevant Gita verse and ARLIAI API."""
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
    
    response = generate_text(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    logger.info(f"Raw AI response for guidance query '{query}':\n{response}")
    
    parts = [part.strip() for part in response.split('000')]
    if len(parts) == 2:
        empathy, explanation = parts
    else:
        empathy = "I understand that you might be feeling uncertain or seeking guidance. Many people experience similar challenges in their lives."
        explanation = response.strip()
    
    transliteration_lines = transliteration.split(' .\n')
    formatted_transliteration = '\n\n'.join(transliteration_lines)
    explanation_paragraphs = [para.strip() for para in explanation.split('\n') if para.strip()]
    formatted_explanation = '<br><br>'.join(explanation_paragraphs) if explanation_paragraphs else explanation
    formatted_shlok = '<br>'.join(shlok.split(' | '))

    formatted_response = (
        empathy + "\n\n"
        + '<blockquote><span style="font-size: 1.2em; font-weight: bold; font-family: \'Noto Sans Devanagari\', \'Mangal\', sans-serif;">' + formatted_shlok + '</span></blockquote>' + "\n\n"
        + formatted_transliteration + "<br><br>"
        + formatted_explanation
    )
    return formatted_response

# Routes
@app.route('/')
def serve_index():
    """Serve the main HTML page."""
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests, either with pre-saved answers or ARLIAI-generated responses."""
    try:
        data = request.json
        query = data.get('message', '')
        user_id = data.get('id', '')
        if not query or not user_id:
            return jsonify({"error": "No message or ID provided"}), 400

        conversation_history[user_id] = {"role": "user", "content": query}

        # Check for pre-saved answers
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
            if is_guidance_query(query):
                formatted_response = get_guidance_response(query)
            else:
                prompt = (
                    f"The user said: '{query}'. Respond in a concise, friendly manner without referencing the Bhagavad Gita. "
                    "Format your response in Markdown for best readability."
                )
                response = generate_text(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1")
                logger.info(f"Raw AI response for non-guidance query '{query}':\n{response}")
                formatted_response = response.strip()

        ai_id = f"ai_{user_id}"
        conversation_history[ai_id] = {"role": "ai", "content": formatted_response}
        return jsonify({"response": formatted_response, "id": ai_id})
    
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {str(e)}")
        fallback_response = "I’m having trouble generating a response right now. Please try again later."
        ai_id = f"ai_{user_id if 'user_id' in locals() else 'unknown'}"
        if 'user_id' in locals():
            conversation_history[ai_id] = {"role": "ai", "content": fallback_response}
        return jsonify({"response": fallback_response, "id": ai_id}), 500

@app.route('/random_verse', methods=['GET'])
def random_verse():
    """Return a random verse from the Bhagavad Gita."""
    random_row = gita_df.sample(n=1).iloc[0]
    verse = random_row['Shloka']
    meaning = random_row['EngMeaning']
    return jsonify({"verse": verse, "meaning": meaning})

@app.route('/update_message', methods=['POST'])
def update_message():
    """Update a user's message in the conversation history."""
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
        logger.error(f"Error in /update_message endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/regenerate_after', methods=['POST'])
def regenerate_after():
    """Regenerate an AI response for a previous user query."""
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
            response = generate_text(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1")
            logger.info(f"Raw AI response for non-guidance query (regenerate) '{query}':\n{response}")
            formatted_response = response.strip()

        ai_id = f"ai_{user_id}"
        conversation_history[ai_id] = {"role": "ai", "content": formatted_response}
        return jsonify({"response": formatted_response, "id": ai_id})

    except Exception as e:
        logger.error(f"Error in /regenerate_after endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/clear', methods=['POST'])
def clear_conversation():
    """Clear the conversation history."""
    try:
        global conversation_history
        conversation_history = {}
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error in /clear endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
