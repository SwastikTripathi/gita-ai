from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os
import logging
import json

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load ARLIAI API key from environment variable
ARLIAI_API_KEY = os.environ.get("ARLIAI_API_KEY", "dae4ebbe-f193-4de3-a929-1e9d557c8554")
API_URL = "https://api.arliai.com/v1/chat/completions"

# Load data files (relative to api/ directory)
gita_df = pd.read_csv('api/bhagwad_gita.csv')
with open('api/pre_saved_answers.json', 'r', encoding='utf-8') as f:
    pre_saved_data = json.load(f)

# Load Sentence Transformer model and compute embeddings
st_model = SentenceTransformer('all-MiniLM-L6-v2')
verse_meanings = gita_df['EngMeaning'].tolist()
verse_embeddings = st_model.encode(verse_meanings, convert_to_tensor=False)
verse_embeddings = np.array(verse_embeddings)
pre_saved_questions = [item['question'] for item in pre_saved_data['questions']]
pre_saved_answers = [item['answer'] for item in pre_saved_data['questions']]
pre_saved_embeddings = st_model.encode(pre_saved_questions, convert_to_tensor=False)
pre_saved_embeddings = np.array(pre_saved_embeddings)

def generate_text(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_new_tokens=500):
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
        return response_json["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise

def get_most_relevant_verse(query):
    query_embedding = st_model.encode([query], convert_to_tensor=False)[0]
    similarities = np.dot(verse_embeddings, query_embedding) / (
        np.linalg.norm(verse_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    most_similar_idx = np.argmax(similarities)
    return gita_df.iloc[most_similar_idx]

def is_guidance_query(query):
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
    parts = [part.strip() for part in response.split('000')]
    empathy = parts[0] if len(parts) == 2 else "I understand you might be seeking clarity. Many face similar moments."
    explanation = parts[1] if len(parts) == 2 else response.strip()
    formatted_shlok = '<br>'.join(shlok.split(' | '))
    formatted_transliteration = '\n\n'.join(transliteration.split(' .\n'))
    formatted_explanation = '<br><br>'.join([p.strip() for p in explanation.split('\n') if p.strip()])
    return (
        f"{empathy}\n\n"
        f'<blockquote><span style="font-size: 1.2em; font-weight: bold; font-family: \'Noto Sans Devanagari\', \'Mangal\', sans-serif;">{formatted_shlok}</span></blockquote>\n\n'
        f"{formatted_transliteration}<br><br>{formatted_explanation}"
    )

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('message', '')
        user_id = data.get('id', '')
        if not query or not user_id:
            return jsonify({"error": "No message or ID provided"}), 400

        query_embedding = st_model.encode([query], convert_to_tensor=False)[0]
        similarities = np.dot(pre_saved_embeddings, query_embedding) / (
            np.linalg.norm(pre_saved_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        max_similarity = np.max(similarities)
        if max_similarity > 0.8:
            most_similar_idx = np.argmax(similarities)
            formatted_response = pre_saved_answers[most_similar_idx]
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
        return jsonify({"response": formatted_response, "id": ai_id})
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        ai_id = f"ai_{user_id if 'user_id' in locals() else 'unknown'}"
        return jsonify({"response": "I’m having trouble responding right now. Try again later.", "id": ai_id}), 500

@app.route('/random_verse', methods=['GET'])
def random_verse():
    random_row = gita_df.sample(n=1).iloc[0]
    verse = random_row['Shloka']
    meaning = random_row['EngMeaning']
    return jsonify({"verse": verse, "meaning": meaning})

@app.route('/regenerate_after', methods=['POST'])
def regenerate_after():
    try:
        data = request.json
        query = data.get('message', '')
        user_id = data.get('id', '')
        if not query:
            return jsonify({"error": "No message provided"}), 400

        if is_guidance_query(query):
            formatted_response = get_guidance_response(query)
        else:
            prompt = (
                f"The user said: '{query}'. Respond in a concise, friendly manner without referencing the Bhagavad Gita. "
                "Format your response in Markdown for best readability."
            )
            formatted_response = generate_text(prompt, max_new_tokens=100)

        ai_id = f"ai_{user_id}"
        return jsonify({"response": formatted_response, "id": ai_id})
    except Exception as e:
        logger.error(f"Error in /regenerate_after: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/clear', methods=['POST'])
def clear_conversation():
    return jsonify({"success": True})

# Vercel handler
def handler(event, context):
    from werkzeug.wrappers import Request
    environ = {
        'REQUEST_METHOD': event['httpMethod'],
        'PATH_INFO': event['path'],
        'QUERY_STRING': '&'.join([f"{k}={v}" for k, v in event.get('queryStringParameters', {}).items()]),
        'CONTENT_TYPE': event.get('headers', {}).get('Content-Type', ''),
        'CONTENT_LENGTH': event.get('headers', {}).get('Content-Length', ''),
        'wsgi.input': event.get('body', '').encode('utf-8') if event.get('body') else b'',
        'wsgi.errors': sys.stderr,
        'wsgi.version': (1, 0),
        'wsgi.url_scheme': 'https',
        'wsgi.multithread': False,
        'wsgi.multiprocess': False,
        'wsgi.run_once': False,
        'SERVER_NAME': 'vercel',
        'SERVER_PORT': '443',
        'SERVER_PROTOCOL': 'HTTP/1.1',
    }
    for key, value in event.get('headers', {}).items():
        environ[f'HTTP_{key.upper().replace("-", "_")}'] = value
    request = Request(environ)
    with app.request_context(request):
        response = app.full_dispatch_request()
    return {
        'statusCode': response.status_code,
        'headers': dict(response.headers),
        'body': response.get_data(as_text=True)
    }

if __name__ == '__main__':
    app.run(debug=True, port=5000)
