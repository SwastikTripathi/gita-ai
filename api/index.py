from flask import Flask, request, jsonify
import pandas as pd
import requests
import os
import logging
import json
from collections import Counter

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

# Load Bhagavad Gita data
gita_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'bhagwad_gita.csv'))
verse_meanings = gita_df['EngMeaning'].tolist()

# Load pre-saved answers (optional, if you still want to use them)
try:
    with open(os.path.join(os.path.dirname(__file__), 'pre_saved_answers.json'), 'r', encoding='utf-8') as f:
        pre_saved_data = json.load(f)
    pre_saved_questions = [item['question'] for item in pre_saved_data['questions']]
    pre_saved_answers = [item['answer'] for item in pre_saved_data['questions']]
except FileNotFoundError:
    logger.warning("pre_saved_answers.json not found, skipping pre-saved answers")
    pre_saved_questions = []
    pre_saved_answers = []

# In-memory conversation storage (non-persistent in serverless)
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
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        generated_text = response_json["choices"][0]["message"]["content"]
        logger.info(f"API response for model '{model}': {generated_text}")
        return generated_text.strip()
    except Exception as e:
        logger.error(f"Error generating text with Arliai API: {str(e)}")
        raise

def get_jaccard_similarity(str1, str2):
    """Compute Jaccard similarity between two strings based on word overlap."""
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0

def get_most_relevant_verse(query):
    """Return the most relevant verse row based on keyword similarity."""
    similarities = [get_jaccard_similarity(query, meaning) for meaning in verse_meanings]
    most_similar_idx = similarities.index(max(similarities))
    return gita_df.iloc[most_similar_idx]

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('message', '')
        user_id = data.get('id', '')
        if not query or not user_id:
            return jsonify({"error": "No message or ID provided"}), 400

        conversation_history[user_id] = {"role": "user", "content": query}
        logger.info(f"Stored user message: {query}, ID: {user_id}")

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
            logger.info(f"Sending prompt to Arliai: {prompt[:100]}...")
            
            response = generate_text(prompt, max_new_tokens=150)
            logger.info(f"Arliai response: {response[:50]}...")
            
            explanation = response.strip()
            formatted_response = (
                f"### Hindi Shlok\n{shloka}\n\n"
                f"### Transliteration\n{transliteration}\n\n"
                f"### Explanation\n{explanation}"
            )
        else:
            # Check pre-saved answers first (if available)
            if pre_saved_questions:
                similarities = [get_jaccard_similarity(query, q) for q in pre_saved_questions]
                max_similarity = max(similarities)
                if max_similarity > 0.5:
                    most_similar_idx = similarities.index(max_similarity)
                    formatted_response = pre_saved_answers[most_similar_idx]
                    logger.info(f"Using pre-saved answer for query '{query}' with similarity {max_similarity}")
                else:
                    prompt = (
                        f"The user said: '{query}'. "
                        f"Respond in a concise, friendly manner without referencing the Bhagavad Gita. "
                        f"Format your response in Markdown."
                    )
                    logger.info(f"Sending prompt to Arliai: {prompt}")
                    response = generate_text(prompt, max_new_tokens=100)
                    logger.info(f"Arliai response: {response}")
                    formatted_response = response.strip()
            else:
                prompt = (
                    f"The user said: '{query}'. "
                    f"Respond in a concise, friendly manner without referencing the Bhagavad Gita. "
                    f"Format your response in Markdown."
                )
                logger.info(f"Sending prompt to Arliai: {prompt}")
                response = generate_text(prompt, max_new_tokens=100)
                logger.info(f"Arliai response: {response}")
                formatted_response = response.strip()

        ai_id = f"ai_{user_id}"
        conversation_history[ai_id] = {"role": "ai", "content": formatted_response}
        logger.info(f"Stored AI response, ID: {ai_id}")

        return jsonify({"response": formatted_response, "id": ai_id})
    
    except Exception as e:
        logger.error(f"Error in /api/chat endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

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
            logger.info(f"Updated message ID {message_id} to: {new_content}")
            return jsonify({"success": True})
        else:
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
        logger.info(f"Regenerating for query: {query}")

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
            logger.info(f"Sending prompt to Arliai: {prompt[:100]}...")
            
            response = generate_text(prompt, max_new_tokens=150)
            logger.info(f"Arliai response: {response[:50]}...")
            
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
            logger.info(f"Sending prompt to Arliai: {prompt}")
            
            response = generate_text(prompt, max_new_tokens=100)
            logger.info(f"Arliai response: {response}")
            
            formatted_response = response.strip()

        ai_id = f"ai_{user_id}"
        conversation_history[ai_id] = {"role": "ai", "content": formatted_response}
        logger.info(f"Stored regenerated AI response, ID: {ai_id}")

        return jsonify({"response": formatted_response, "id": ai_id})

    except Exception as e:
        logger.error(f"Error in /api/regenerate_after endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    try:
        global conversation_history
        conversation_history = {}
        logger.info("Conversation history cleared")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error in /api/clear endpoint: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
