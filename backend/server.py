"""
EightySix Chemistry - Production Server
Cloudflare R2 + Semantic Search
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import requests
import json
import os
import re

app = Flask(__name__)
CORS(app)

# Environment Configuration
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', 'your-key-here')
R2_BUCKET_URL = os.environ.get('R2_BUCKET_URL', 'https://pub-xxxxx.r2.dev')
PORT = int(os.environ.get('PORT', 5000))
PRODUCTION = os.environ.get('PRODUCTION', 'false').lower() == 'true'

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get('MODEL', 'google/gemini-2.0-flash-exp:free')

# Book Library
BOOK_LIBRARY = {
    'zumdahl': {
        'name': 'General Chemistry',
        'author': 'Zumdahl & Zumdahl',
        'chunks_url': f'{R2_BUCKET_URL}/zumdahl_chunks_with_embeddings.json'
    },
    'atkins': {
        'name': 'Physical Chemistry',
        'author': 'Atkins & de Paula',
        'chunks_url': f'{R2_BUCKET_URL}/atkins_chunks_with_embeddings.json'
    },
    'harris': {
        'name': 'Quantitative Chemical Analysis',
        'author': 'Daniel C. Harris',
        'chunks_url': f'{R2_BUCKET_URL}/harris_chunks_with_embeddings.json'
    },
    'klein': {
        'name': 'Organic Chemistry',
        'author': 'David Klein',
        'chunks_url': f'{R2_BUCKET_URL}/klein_chunks_with_embeddings.json'
    }
}

current_book_info = {'id': None, 'name': 'No book loaded', 'author': ''}

# Semantic Search Engine
class SemanticSearch:
    def __init__(self):
        print("\n🤖 Initializing Semantic Search...")
        print("📥 Loading model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model ready!\n")
        self.chunks = []
        self.textbook = None
    
    def load_chunks_from_r2(self, url):
        try:
            print(f"\n📥 Fetching from R2: {url}")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            chunks = response.json()
            self.chunks = chunks
            self.textbook = {'pages': chunks}
            print(f"✅ Loaded {len(chunks)} chunks!")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def search(self, question):
        if not self.chunks:
            return "No textbook loaded.", 0.0, False
        
        question_embedding = self.model.encode(question)
        
        scored = []
        for chunk in self.chunks:
            similarity = util.cos_sim(question_embedding, chunk['embedding']).item()
            scored.append((chunk, similarity))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        top_chunk, top_score = scored[0]
        is_relevant = top_score >= 0.35
        
        context = "\n\n".join([
            f"[Page {chunk['page']}] {chunk['text']}"
            for chunk, _ in scored[:3]
        ])
        
        return context, top_score, is_relevant
    
    def get_candidate_pages(self, topic, top_k=5):
        if not self.chunks:
            return []
        
        topic_embedding = self.model.encode(topic)
        
        scored = []
        for chunk in self.chunks:
            similarity = util.cos_sim(topic_embedding, chunk['embedding']).item()
            scored.append({
                'page': chunk['page'],
                'text': chunk['text'],
                'score': similarity
            })
        
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]

search_engine = SemanticSearch()

# AI Helper
def call_ai(prompt, system_prompt="You are an expert chemistry tutor."):
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 3000
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# Routes
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name': 'EightySix Chemistry API',
        'version': '2.0',
        'status': 'running',
        'search': 'semantic'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'textbook_loaded': search_engine.textbook is not None,
        'chunks_count': len(search_engine.chunks),
        'current_book': current_book_info,
        'semantic_search': True
    })

@app.route('/load-book', methods=['POST'])
def load_book():
    global current_book_info
    try:
        book_id = request.json.get('bookId')
        
        if book_id not in BOOK_LIBRARY:
            return jsonify({'success': False, 'error': 'Book not found'}), 404
        
        book = BOOK_LIBRARY[book_id]
        success = search_engine.load_chunks_from_r2(book['chunks_url'])
        
        if not success:
            return jsonify({'success': False, 'error': 'Failed to load'}), 500
        
        current_book_info = {
            'id': book_id,
            'name': book['name'],
            'author': book['author']
        }
        
        return jsonify({
            'success': True,
            'book_id': book_id,
            'book_name': book['name'],
            'author': book['author'],
            'chunks_count': len(search_engine.chunks)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get-library', methods=['GET'])
def get_library():
    books = []
    for book_id, info in BOOK_LIBRARY.items():
        books.append({
            'id': book_id,
            'name': info['name'],
            'author': info['author'],
            'available': True
        })
    return jsonify({'success': True, 'books': books, 'current_book': current_book_info})

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json.get('question', '')
        complexity = request.json.get('complexity', 3)
        
        if not search_engine.chunks:
            return jsonify({'success': False, 'error': 'No textbook loaded'}), 400
        
        context, score, is_relevant = search_engine.search(question)
        
        complexity_map = {
            1: "Explain simply for beginners.",
            2: "Clear high school level.",
            3: "Balanced with proper terms.",
            4: "Detailed for advanced students.",
            5: "University level."
        }
        
        instruction = complexity_map.get(complexity, complexity_map[3])
        
        if is_relevant:
            prompt = f"""Chemistry tutor answering from textbook.

CONTEXT:
{context}

QUESTION: {question}

{instruction}

Answer based on context."""
        else:
            prompt = f"Student asked: '{question}' - Not in textbook. Politely redirect."
        
        answer = call_ai(prompt)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'context': context,
            'score': float(score),
            'is_relevant': is_relevant
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/generate-flashcards', methods=['POST'])
def generate_flashcards():
    try:
        topic = request.json.get('topic', '')
        count = request.json.get('count', 5)
        
        if not search_engine.chunks:
            return jsonify({'success': False, 'error': 'No textbook'}), 400
        
        pages = search_engine.get_candidate_pages(topic, top_k=3)
        
        if not pages or pages[0]['score'] < 0.2:
            return jsonify({'success': False, 'error': 'Topic not found'}), 404
        
        content = "\n\n".join([p['text'][:1000] for p in pages])
        
        prompt = f"""Create {count} flashcards about {topic}:

{content}

Format:
Q: [Question]
A: [Answer]"""
        
        response = call_ai(prompt)
        
        flashcards = []
        lines = response.split('\n')
        current_q = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                current_q = line[2:].strip()
            elif line.startswith('A:') and current_q:
                flashcards.append({'front': current_q, 'back': line[2:].strip()})
                current_q = None
                if len(flashcards) >= count:
                    break
        
        return jsonify({'success': True, 'flashcards': flashcards})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🧪 EIGHTYSIX CHEMISTRY")
    print("=" * 60)
    print(f"🔍 Search: Semantic (sentence-transformers)")
    print(f"☁️  Storage: Cloudflare R2")
    print(f"🔑 API: {'✅' if OPENROUTER_API_KEY != 'your-key-here' else '❌'}")
    print(f"📦 R2: {'✅' if R2_BUCKET_URL != 'https://pub-xxxxx.r2.dev' else '❌'}")
    print(f"🌐 Port: {PORT}")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=not PRODUCTION)
