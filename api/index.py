# api/index.py
from flask import Flask, request, jsonify
import pandas as pd
import os
import logging
import json
import string
from huggingface_hub import InferenceClient

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load Hugging Face API key from environment variable
HF_API_KEY = os.environ.get("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY environment variable is not set")
client = InferenceClient(token=HF_API_KEY)

# Tokenization and Jaccard similarity functions
def tokenize(text):
    """Tokenize text into a set of lowercase words, removing punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return set(text.split())

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets of words."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Load data with absolute paths
BASE_DIR = os.path.dirname(__file__)
try:
    gita_df = pd.read_csv(os.path.join(BASE_DIR, 'bhagwad_gita.csv'))
    with open(os.path.join(BASE_DIR, 'pre_saved_answers.json'), 'r', encoding='utf-8') as f:
        pre_saved_data = json.load(f)
    pre_saved_questions = [item['question'] for item in pre_saved_data['questions']]
    pre_saved_answers = [item['answer'] for item in pre_saved_data['questions']]
except FileNotFoundError as e:
    logger.error(f"Error loading data files: {str(e)}")
    raise

# Pre-tokenize pre-saved questions and verse meanings
pre_saved_question_sets = [tokenize(question) for question in pre_saved_questions]
verse_meaning_sets = [tokenize(meaning) for meaning in gita_df['EngMeaning']]

# In-memory conversation storage
conversation_history = {}

def generate_text(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_new_tokens=256):
    """Calls the Hugging Face Inference API and returns the generated text."""
    try:
        response = client.text_generation(
            prompt=prompt,
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
            timeout=90  # Add a timeout of 30 seconds
        )
        logger.info(f"HF API response for model '{model}': {response}")
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating text with Hugging Face API: {str(e)}")
        fallback_message = (
            "Oops, response timed out. As this is a prototype, sometimes a large number of users delay the responses. "
            "Please try some time later. You can also support this app from the coffee button on the bottom right."
        )
        return fallback_message  # Return fallback instead of raising an exception

def get_most_relevant_verse(query):
    """Return the most relevant verse based on keyword matching."""
    query_set = tokenize(query)
    similarities = [jaccard_similarity(query_set, v_set) for v_set in verse_meaning_sets]
    most_similar_idx = similarities.index(max(similarities)) if similarities else 0
    return gita_df.iloc[most_similar_idx]

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
    
    response = generate_text(prompt, max_new_tokens=256)
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

        query_set = tokenize(query)
        similarities = [jaccard_similarity(query_set, q_set) for q_set in pre_saved_question_sets]
        max_similarity = max(similarities) if similarities else 0
        if max_similarity > 0.5:
            most_similar_idx = similarities.index(max_similarity)
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
                formatted_response = generate_text(prompt, max_new_tokens=100)

        ai_id = f"ai_{user_id}"
        conversation_history[ai_id] = {"role": "ai", "content": formatted_response}
        return jsonify({"response": formatted_response, "id": ai_id})

    except Exception as e:
        logger.error(f"Error in /api/chat endpoint: {str(e)}")
        fallback_response = (
            "Oops, response timed out. As this is a prototype, sometimes a large number of users delay the responses. "
            "Please try some time later. You can also support this app from the coffee button on the bottom right."
        )
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
            formatted_response = generate_text(prompt, max_new_tokens=100)

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

# Serve static files (e.g., index.html)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    if not path or path == 'index.html':
        return app.send_static_file('index.html')
    return app.send_static_file(path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
