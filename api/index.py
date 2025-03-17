from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
import os

# Initialize Flask app
app = Flask(__name__)

# Load Hugging Face API key securely from an environment variable
HF_API_KEY = os.environ.get("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY environment variable is not set")
client = InferenceClient(token=HF_API_KEY)

# Load Bhagavad Gita data and precomputed embeddings
gita_df = pd.read_csv('api/bhagwad_gita.csv')
verse_embeddings = np.load('api/verse_embeddings.npy')

# In-memory conversation storage (for production, consider a persistent DB)
conversation_history = {}

def get_most_relevant_verse(query):
    """Return the most relevant verse row from the Bhagavad Gita based on the query."""
    query_embedding = client.feature_extraction(
        text=query,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Flatten to ensure (384,) shape
    query_embedding = np.array(query_embedding).flatten()
    if query_embedding.shape[0] != 384:
        raise ValueError(f"Expected query embedding shape (384,), got {query_embedding.shape}")
    
    similarities = np.dot(verse_embeddings, query_embedding) / (
        np.linalg.norm(verse_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
    )
    most_similar_idx = np.argmax(similarities)
    return gita_df.iloc[most_similar_idx]

def is_guidance_query(query):
    """Determine if the user's query is seeking guidance, advice, or wisdom."""
    guidance_keywords = ['guidance', 'insight', 'help', 'advice', 'confused', 'lost', 'question']
    if any(keyword in query.lower() for keyword in guidance_keywords) or query.strip().endswith('?'):
        return True
    return False

def get_guidance_response(query):
    relevant_verse = get_most_relevant_verse(query)
    shloka = relevant_verse['Shloka']
    transliteration = relevant_verse['Transliteration']
    
    prompt = (
        f"User query: {query}\n\n"
        "Please respond in Markdown with the following structure:\n\n"
        "1. Begin with one or two empathetic lines acknowledging that many feel similar emotions, without repeating the user's query.\n"
        "2. On a new line, display the following Hindi shlok as a bold, indented block (using a blockquote) with no label. Format it as:\n"
        "   > **[Hindi Shlok]**\n"
        f"   **{shloka}**\n\n"
        "3. Next, output the transliteration on two separate lines. If the transliteration text contains a vertical bar ('|'), split the text at the vertical bar so that the part before the bar appears on the first line and the part after appears on the second line. Do not include any label before the transliteration.\n"
        "4. Insert two line breaks after the transliteration.\n"
        "5. Finally, provide a detailed explanation in one to three paragraphs that explains the context of the verse, what it is trying to say, and how it relates to the user's query. Use modern-day examples to reinforce the teaching. Ensure that each paragraph is separated by two line breaks.\n\n"
        "Do not include any horizontal lines or additional formatting markers."
    )
    
    response = client.text_generation(
        prompt=prompt,
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    explanation = response.strip()
    return explanation

@app.route('/')
def serve_index():
    return send_from_directory('../static', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('message', '')
        user_id = data.get('id', '')
        if not query or not user_id:
            return jsonify({"error": "No message or ID provided"}), 400

        conversation_history[user_id] = {"role": "user", "content": query}

        if is_guidance_query(query):
            formatted_response = get_guidance_response(query)
        else:
            prompt = (
                f"The user said: '{query}'. Respond in a concise, friendly manner without referencing the Bhagavad Gita. "
                "Format your response in Markdown for best readability."
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

        ai_id = f"ai_{user_id}"
        conversation_history[ai_id] = {"role": "ai", "content": formatted_response}
        return jsonify({"response": formatted_response, "id": ai_id})
    
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return jsonify({"error": "Something went wrong on the server"}), 500

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
            response = client.text_generation(
                prompt=prompt,
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            formatted_response = response.strip()

        ai_id = f"ai_{user_id}"
        conversation_history[ai_id] = {"role": "ai", "content": formatted_response}
        return jsonify({"response": formatted_response, "id": ai_id})

    except Exception as e:
        print(f"Error in regenerate_after: {str(e)}")
        return jsonify({"error": "Something went wrong on the server"}), 500

@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    try:
        global conversation_history
        conversation_history = {}
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error in clear_conversation: {str(e)}")
        return jsonify({"error": "Something went wrong on the server"}), 500
