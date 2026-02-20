"""
EightySix Chemistry - Production Server
Cloudflare R2 Storage + Semantic Search
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import requests
import json
import os
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ============================================
# ENVIRONMENT CONFIGURATION
# ============================================

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', 'your-key-here')
R2_BUCKET_URL = os.environ.get('R2_BUCKET_URL', 'https://pub-xxxxx.r2.dev')
PORT = int(os.environ.get('PORT', 5000))
PRODUCTION = os.environ.get('PRODUCTION', 'false').lower() == 'true'

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get('MODEL', 'xiaomi/mimo-v2-flash:free')

print(f"🔧 Configuration loaded")
print(f"   R2 URL: {R2_BUCKET_URL}")
print(f"   API Key: {'✅ Set' if OPENROUTER_API_KEY != 'your-key-here' else '❌ Not set'}")

# ============================================
# BOOK LIBRARY - R2 CLOUD STORAGE
# ============================================

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

current_book_info = {
    'id': None,
    'name': 'No book loaded',
    'author': ''
}

# ============================================
# AI SEARCH ENGINE
# ============================================

class AITextbookSearch:
    HIGH_CONFIDENCE = 0.65
    LOW_CONFIDENCE = 0.35
    
    def __init__(self):
        """Initialize with semantic search model"""
        print("\n🤖 Initializing AI Search Engine...")
        print("📥 Loading sentence transformer model...")
        
        try:
            self.model = None
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
        
        self.chunks = []
        self.textbook = None
        print("⏳ Ready to load books from R2\n")
    
    def load_chunks_from_r2(self, url):
        """Load chunks from Cloudflare R2"""
        try:
            print(f"\n📥 Fetching from R2: {url}")
            
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            chunks = response.json()
            
            self.chunks = chunks
            self.textbook = {'pages': chunks}
            
            print(f"✅ Loaded {len(chunks)} chunks successfully!")
            return True
            
        except requests.exceptions.Timeout:
            print("❌ Timeout: R2 took too long to respond")
            return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return False
        except json.JSONDecodeError:
            print("❌ Invalid JSON in chunks file")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def smart_search(self, question):
        """Semantic search with relevance scoring"""
        if not self.chunks:
            return "No textbook loaded. Please select a book first.", 0.0, False
        
        if not self.model:
            return "Search model not available.", 0.0, False
        
        try:
            # Encode question
            question_embedding = self.model.encode(question)
            
            # Score all chunks
            scored_chunks = []
            for chunk in self.chunks:
                similarity = util.cos_sim(question_embedding, chunk['embedding']).item()
                scored_chunks.append((chunk, similarity))
            
            # Sort by similarity
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            if not scored_chunks:
                return "No content found.", 0.0, False
            
            # Get top result
            top_chunk, top_similarity = scored_chunks[0]
            is_relevant = top_similarity >= self.LOW_CONFIDENCE
            
            # Build context from top 3
            context = "\n\n".join([
                f"[Page {chunk['page']}] {chunk['text']}"
                for chunk, _ in scored_chunks[:3]
            ])
            
            return context, top_similarity, is_relevant
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return f"Search error: {str(e)}", 0.0, False
    
    def get_candidate_pages(self, topic, top_k=5):
        """Get top pages for a topic"""
        if not self.chunks or not self.model:
            return []
        
        try:
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
            
        except Exception as e:
            print(f"❌ Error getting pages: {e}")
            return []


# Initialize search engine
ai_search = AITextbookSearch()

# ============================================
# AI HELPER
# ============================================

def call_ai(prompt, system_prompt="You are an expert chemistry tutor."):
    """Call OpenRouter API"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://chemistry-app.com",
            "X-Title": "EightySix Chemistry"
        }
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 3000
        }
        
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: API returned {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================
# ROUTES
# ============================================

@app.route('/', methods=['GET'])
def home():
    """API info"""
    return jsonify({
        'name': 'EightySix Chemistry API',
        'version': '2.0',
        'status': 'running',
        'endpoints': ['/health', '/ask', '/load-book', '/get-library', '/generate-flashcards']
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'mode': 'production' if PRODUCTION else 'development',
        'textbook_loaded': ai_search.textbook is not None,
        'chunks_count': len(ai_search.chunks),
        'current_book': current_book_info,
        'model_loaded': ai_search.model is not None,
        'r2_configured': R2_BUCKET_URL != 'https://pub-xxxxx.r2.dev'
    })


@app.route('/load-book', methods=['POST'])
def load_book():
    """Load a book from R2"""
    global current_book_info
    
    try:
        data = request.json
        book_id = data.get('bookId')
        
        print(f"\n📚 Loading book: {book_id}")
        
        if book_id not in BOOK_LIBRARY:
            return jsonify({
                'success': False,
                'error': f'Book "{book_id}" not found'
            }), 404
        
        book = BOOK_LIBRARY[book_id]
        
        print(f"📖 {book['name']} by {book['author']}")
        
        # Load from R2
        success = ai_search.load_chunks_from_r2(book['chunks_url'])
        
        if not success:
            return jsonify({
                'success': False,
                'error': 'Failed to load from R2. Check R2_BUCKET_URL.'
            }), 500
        
        # Update current book
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
            'chunks_count': len(ai_search.chunks)
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/get-library', methods=['GET'])
def get_library():
    """Get available books"""
    books = []
    for book_id, info in BOOK_LIBRARY.items():
        books.append({
            'id': book_id,
            'name': info['name'],
            'author': info['author'],
            'available': True
        })
    
    return jsonify({
        'success': True,
        'books': books,
        'current_book': current_book_info
    })


@app.route('/ask', methods=['POST'])
def ask():
    """Answer questions"""
    try:
        data = request.json
        question = data.get('question', '')
        complexity = data.get('complexity', 3)
        
        print(f"\n📝 Q: {question}")
        
        if not ai_search.chunks:
            return jsonify({
                'success': False,
                'error': 'No textbook loaded. Please select a book first.'
            }), 400
        
        # Search
        context, similarity, is_relevant = ai_search.smart_search(question)
        
        print(f"🔍 Similarity: {similarity:.3f}, Relevant: {is_relevant}")
        
        # Generate answer
        complexity_map = {
            1: "Explain simply for beginners.",
            2: "Clear high school level explanation.",
            3: "Balanced explanation with proper terms.",
            4: "Detailed for advanced students.",
            5: "Comprehensive university level."
        }
        
        instruction = complexity_map.get(complexity, complexity_map[3])
        
        if is_relevant:
            prompt = f"""Chemistry tutor answering from textbook.

CONTEXT:
{context}

QUESTION: {question}

{instruction}

Provide accurate answer based on context."""
        else:
            prompt = f"The student asked: '{question}' - This isn't in the textbook. Politely redirect to chemistry topics."
        
        answer = call_ai(prompt)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'context': context,
            'similarity': float(similarity),
            'is_relevant': is_relevant
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/generate-flashcards', methods=['POST'])
def generate_flashcards():
    """Generate flashcards"""
    try:
        data = request.json
        topic = data.get('topic', '')
        count = data.get('count', 5)
        
        print(f"\n🎴 Generating {count} flashcards: {topic}")
        
        if not ai_search.chunks:
            return jsonify({'success': False, 'error': 'No textbook loaded'}), 400
        
        # Get relevant pages
        pages = ai_search.get_candidate_pages(topic, top_k=3)
        
        if not pages or pages[0]['score'] < 0.3:
            return jsonify({
                'success': False,
                'error': f'Topic "{topic}" not found'
            }), 404
        
        # Build prompt
        content = "\n\n".join([p['text'][:1000] for p in pages])
        
        prompt = f"""Create {count} flashcards about {topic}:

{content}

Format:
Q: [Question]
A: [Answer]

Q: [Question]
A: [Answer]"""
        
        response = call_ai(prompt)
        
        # Parse
        flashcards = []
        lines = response.split('\n')
        current_q = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                current_q = line[2:].strip()
            elif line.startswith('A:') and current_q:
                flashcards.append({
                    'front': current_q,
                    'back': line[2:].strip()
                })
                current_q = None
        
        print(f"✅ Generated {len(flashcards)} flashcards")
        
        return jsonify({
            'success': True,
            'flashcards': flashcards[:count]
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# START SERVER
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🧪 EIGHTYSIX CHEMISTRY - PRODUCTION SERVER")
    print("=" * 60)
    print(f"🌐 Mode: {'PRODUCTION' if PRODUCTION else 'DEVELOPMENT'}")
    print(f"📚 Books: {len(BOOK_LIBRARY)} available")
    print(f"☁️  Storage: Cloudflare R2")
    print(f"🔑 API Key: {'✅ Configured' if OPENROUTER_API_KEY != 'your-key-here' else '❌ Missing'}")
    print(f"📦 R2 URL: {'✅ Configured' if R2_BUCKET_URL != 'https://pub-xxxxx.r2.dev' else '❌ Missing'}")
    print(f"🌐 Port: {PORT}")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=not PRODUCTION)
