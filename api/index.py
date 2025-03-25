from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import os
import logging
import json
from verse_embeddings import verse_embeddings, gita_df  # Import precomputed data

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

# Load Sentence Transformer model for query encoding (not embeddings)
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load pre-saved answers
try:
    with open('api/pre_saved_answers.json', 'r', encoding='utf-8') as f:
        pre_saved_data = json.load(f)
    pre_saved_questions = [item['question'] for item in pre_saved_data['questions']]
    pre_saved_answers = [item['answer'] for item in pre_saved_data['questions']]
    pre_saved_embeddings = st_model.encode(pre_saved_questions, convert_to_tensor=False)
    pre_saved_embeddings = np.array(pre_saved_embeddings)
except FileNotFoundError:
    logger.error("pre_saved_answers.json not found in api/ directory")
    raise
except Exception as e:
    logger.error(f"Error loading pre-saved answers: {str(e)}")
    raise

# In-memory conversation storage (non-persistent in Vercel)
conversation_history = {}

def generate_text(prompt, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    """Calls the Hugging Face Inference API and returns the generated text."""
    try:
        response = client.text_generation(
            prompt=prompt,
            model=model,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        logger.info(f"API response for model '{model}': {response[:50]}...")
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating text with Hugging Face API: {str(e)}")
        raise

def get_most_relevant_verse(query):
    """Return the most relevant verse row from the Bhagavad Gita based on the query."""
    try:
        query_embedding = st_model.encode([query], convert_to_tensor=False)[0]
        similarities = np.dot(verse_embeddings, query_embedding) / (
            np.linalg.norm(verse_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        most_similar_idx = np.argmax(similarities)
        return gita_df.iloc[most_similar_idx]
    except Exception as e:
        logger.error(f"Error in get_most_relevant_verse: {str(e)}")
        raise

def is_guidance_query(query):
    """Determine if the user's query is seeking guidance, advice, or wisdom."""
    guidance_keywords = ['guidance', 'insight', 'help', 'advice', 'confused', 'lost', 'question']
    return any(keyword in query.lower() for keyword in guidance_keywords) or query.strip().endswith('?')

def get_guidance_response(query):
    """Generate a guidance response with empathy and verse explanation."""
    try:
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
        
        response = generate_text(prompt)
        logger.info(f"Raw AI response for guidance query '{query}':\n{response[:100]}...")
        
        parts = [part.strip() for part in response.split('000')]
        empathy = parts[0] if len(parts) == 2 else "I understand that you might be feeling uncertain or seeking guidance. Many people experience similar challenges in their lives."
        explanation = parts[1] if len(parts) == 2 else response.strip()
        
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
    except Exception as e:
        logger.error(f"Error in get_guidance_response: {str(e)}")
        raise

@app.route('/')
def serve_index():
    try:
        return send_from_directory('../static', 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return jsonify({"error": "Failed to serve the page"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('message', '')
        user_id = data.get('id', '')
        if not query or not user_id:
            return jsonify({"error": "No message or ID provided"}), 400

        conversation_history[user_id] = {"role": "user", "content": query}

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
                response = generate_text(prompt, max_new_tokens=100)
                logger.info(f"Raw AI response for non-guidance query '{query}':\n{response}")
                formatted_response = response.strip()

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
    try:
        random_row = gita_df.sample(n=1).iloc[0]
        verse = random_row['Shloka']
        meaning = random_row['EngMeaning']
        return jsonify({"verse": verse, "meaning": meaning})
    except Exception as e:
        logger.error(f"Error in /api/random_verse endpoint: {str(e)}")
        return jsonify({"error": "Failed to fetch a random verse"}), 500

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
            response = generate_text(prompt, max_new_tokens=100)
            logger.info(f"Raw AI response for non-guidance query (regenerate) '{query}':\n{response}")
            formatted_response = response.strip()

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
