"""
EightySix Chemistry - Ultra-Lightweight Production Server
OpenRouter AI for Everything (No Local Models!)
Cloudflare R2 Storage
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os
import re

app = Flask(__name__)
CORS(app)

# ============================================
# CONFIGURATION
# ============================================

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', 'your-key-here')
R2_BUCKET_URL = os.environ.get('R2_BUCKET_URL', 'https://pub-xxxxx.r2.dev')
PORT = int(os.environ.get('PORT', 5000))
PRODUCTION = os.environ.get('PRODUCTION', 'false').lower() == 'true'

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get('MODEL', 'google/gemini-2.0-flash-exp:free')

print(f"🔧 Lightweight Server Configuration")
print(f"   R2: {R2_BUCKET_URL}")
print(f"   API: {'✅ Set' if OPENROUTER_API_KEY != 'your-key-here' else '❌ Not set'}")
print(f"   Mode: Ultra-Lightweight (OpenRouter AI only)")

# ============================================
# BOOK LIBRARY
# ============================================

BOOK_LIBRARY = {
    'zumdahl': {
        'name': 'General Chemistry',
        'author': 'Zumdahl & Zumdahl',
        'chunks_url': f'{R2_BUCKET_URL}/data/chunks_with_embeddings.json'
    },
    'atkins': {
        'name': 'Physical Chemistry',
        'author': 'Atkins & de Paula',
        'chunks_url': f'{R2_BUCKET_URL}/data/atkins_chunks_with_embeddings.json'
    },
    'harris': {
        'name': 'Quantitative Chemical Analysis',
        'author': 'Daniel C. Harris',
        'chunks_url': f'{R2_BUCKET_URL}/data/harris_chunks_with_embeddings.json'
    },
    'klein': {
        'name': 'Organic Chemistry',
        'author': 'David Klein',
        'chunks_url': f'{R2_BUCKET_URL}/data/klein_chunks_with_embeddings.json'
    }
}

current_book_info = {'id': None, 'name': 'No book loaded', 'author': ''}

# ============================================
# INTELLIGENT SEARCH ENGINE
# ============================================

class IntelligentSearch:
    """Uses OpenRouter AI for intelligent semantic search"""
    
    def __init__(self):
        print("\n🤖 Initializing Intelligent Search Engine...")
        print("✅ Using OpenRouter AI for embeddings")
        print("✅ No local model needed!")
        self.chunks = []
        self.textbook = None
        print("⏳ Ready to load books from R2\n")
    
    def load_chunks_from_r2(self, url):
        """Load chunks from Cloudflare R2"""
        try:
            print(f"\n📥 Fetching from R2: {url}")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            chunks = response.json()
            
            # Remove embeddings - we don't need them!
            print(f"🗑️  Removing embeddings to save memory...")
            for chunk in chunks:
                if 'embedding' in chunk:
                    del chunk['embedding']
            
            self.chunks = chunks
            self.textbook = {'pages': chunks}
            
            print(f"✅ Loaded {len(chunks)} chunks (lightweight!)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading from R2: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def ai_search(self, question):
        """Use OpenRouter AI to find relevant chunks"""
        if not self.chunks:
            return "No textbook loaded.", 0.0, False
        
        try:
            # Get a small sample of chunks to search through
            # We'll ask AI to help us find the most relevant ones
            
            # Strategy: Use keyword matching first to narrow down,
            # then use AI to pick the best ones
            
            question_lower = question.lower()
            question_words = set(re.findall(r'\b\w{3,}\b', question_lower))
            
            # Quick keyword pre-filter
            scored = []
            for chunk in self.chunks:
                text_lower = chunk['text'].lower()
                text_words = set(re.findall(r'\b\w{3,}\b', text_lower))
                
                common = question_words & text_words
                
                # Boost for exact phrase matches
                boost = sum(0.3 for word in question_words if len(word) > 4 and word in text_lower)
                
                score = (len(common) / max(len(question_words), 1)) + boost
                scored.append((chunk, score))
            
            # Sort and get top 10 candidates
            scored.sort(key=lambda x: x[1], reverse=True)
            top_candidates = scored[:10]
            
            if not top_candidates or top_candidates[0][1] == 0:
                return "No relevant content found.", 0.0, False
            
            # Use AI to pick the best 3 from top 10
            candidates_text = "\n\n".join([
                f"[Candidate {i+1}, Page {chunk['page']}]\n{chunk['text'][:500]}"
                for i, (chunk, _) in enumerate(top_candidates[:5])
            ])
            
            ai_prompt = f"""Given this question: "{question}"

Which of these textbook excerpts is most relevant? Pick the top 3 and rank them.

{candidates_text}

Respond with just the numbers: "1, 3, 5" (most relevant first)"""
            
            try:
                ai_ranking = call_ai(ai_prompt, "You are a chemistry textbook expert.")
                # Parse the ranking (e.g., "1, 3, 2")
                numbers = [int(n.strip()) for n in re.findall(r'\d+', ai_ranking)]
                
                # Build context from AI-selected chunks
                context_chunks = []
                for num in numbers[:3]:
                    if 1 <= num <= len(top_candidates):
                        context_chunks.append(top_candidates[num-1][0])
                
                if context_chunks:
                    context = "\n\n".join([
                        f"[Page {chunk['page']}] {chunk['text']}"
                        for chunk in context_chunks
                    ])
                    # Score based on AI's confidence
                    score = top_candidates[0][1]
                    is_relevant = score >= 0.15
                    return context, score, is_relevant
                    
            except:
                # Fallback: just use keyword scores
                pass
            
            # Default: return top 3 by keyword score
            context = "\n\n".join([
                f"[Page {chunk['page']}] {chunk['text']}"
                for chunk, _ in top_candidates[:3]
            ])
            
            top_score = top_candidates[0][1]
            is_relevant = top_score >= 0.15
            
            return context, top_score, is_relevant
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return f"Search error: {str(e)}", 0.0, False
    
    def get_candidate_pages(self, topic, top_k=5):
        """Get top pages for a topic"""
        if not self.chunks:
            return []
        
        try:
            topic_lower = topic.lower()
            topic_words = set(re.findall(r'\b\w{3,}\b', topic_lower))
            
            scored = []
            for chunk in self.chunks:
                text_lower = chunk['text'].lower()
                text_words = set(re.findall(r'\b\w{3,}\b', text_lower))
                
                common = topic_words & text_words
                score = len(common) / max(len(topic_words), 1)
                
                # Boost for topic appearing in text
                if topic_lower in text_lower:
                    score += 0.5
                
                scored.append({
                    'page': chunk['page'],
                    'text': chunk['text'],
                    'score': score
                })
            
            scored.sort(key=lambda x: x['score'], reverse=True)
            return scored[:top_k]
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return []

search_engine = IntelligentSearch()

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
            print(f"⚠️  API error: {response.status_code}")
            return f"Error: API returned {response.status_code}"
            
    except Exception as e:
        print(f"❌ AI call error: {e}")
        return f"Error: {str(e)}"

# ============================================
# ROUTES
# ============================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name': 'EightySix Chemistry API',
        'version': '2.0 Ultra-Lightweight',
        'status': 'running',
        'search_type': 'AI-powered (OpenRouter)',
        'memory_footprint': 'minimal',
        'endpoints': {
            'health': '/health',
            'library': '/get-library',
            'load_book': '/load-book',
            'ask': '/ask',
            'flashcards': '/generate-flashcards'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'mode': 'production' if PRODUCTION else 'development',
        'textbook_loaded': search_engine.textbook is not None,
        'chunks_count': len(search_engine.chunks),
        'current_book': current_book_info,
        'search_type': 'AI-powered',
        'local_model': False,
        'r2_configured': R2_BUCKET_URL != 'https://pub-xxxxx.r2.dev'
    })


@app.route('/load-book', methods=['POST'])
def load_book():
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
        
        success = search_engine.load_chunks_from_r2(book['chunks_url'])
        
        if not success:
            return jsonify({
                'success': False,
                'error': 'Failed to load from R2'
            }), 500
        
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
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
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
    
    return jsonify({
        'success': True,
        'books': books,
        'current_book': current_book_info
    })


@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question', '')
        complexity = data.get('complexity', 3)
        
        print(f"\n📝 Question: {question}")
        
        if not search_engine.chunks:
            return jsonify({
                'success': False,
                'error': 'No textbook loaded. Select a book first.'
            }), 400
        
        # AI-powered search
        context, score, is_relevant = search_engine.ai_search(question)
        
        print(f"🔍 Relevance score: {score:.3f}")
        
        # Generate answer
        complexity_map = {
            1: "Explain simply for beginners with everyday language.",
            2: "Clear high school level explanation.",
            3: "Balanced explanation with proper chemistry terms.",
            4: "Detailed explanation for advanced students.",
            5: "Comprehensive university-level explanation."
        }
        
        instruction = complexity_map.get(complexity, complexity_map[3])
        
        if is_relevant:
            prompt = f"""You are a chemistry tutor. Answer based on this textbook content.

TEXTBOOK CONTENT:
{context}

STUDENT'S QUESTION:
{question}

INSTRUCTIONS:
{instruction}

Provide an accurate, helpful answer based on the textbook content above."""
        else:
            prompt = f"""The student asked: "{question}"

This doesn't appear to be covered in the current textbook. Politely let them know and suggest they ask about chemistry topics from the book."""
        
        answer = call_ai(prompt)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'context': context if is_relevant else '',
            'relevance_score': float(score),
            'is_relevant': is_relevant
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/generate-flashcards', methods=['POST'])
def generate_flashcards():
    try:
        data = request.json
        topic = data.get('topic', '')
        count = data.get('count', 5)
        
        print(f"\n🎴 Generating {count} flashcards: {topic}")
        
        if not search_engine.chunks:
            return jsonify({'success': False, 'error': 'No textbook loaded'}), 400
        
        pages = search_engine.get_candidate_pages(topic, top_k=3)
        
        if not pages or pages[0]['score'] < 0.2:
            return jsonify({
                'success': False,
                'error': f'Topic "{topic}" not found in textbook'
            }), 404
        
        content = "\n\n".join([p['text'][:1000] for p in pages])
        
        prompt = f"""Create {count} study flashcards about {topic} from this textbook content:

{content}

Format each flashcard EXACTLY as:
Q: [Clear, specific question]
A: [Concise but complete answer]

Create exactly {count} flashcards covering the key concepts."""
        
        response = call_ai(prompt)
        
        # Parse flashcards
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
                if len(flashcards) >= count:
                    break
        
        print(f"✅ Generated {len(flashcards)} flashcards")
        
        return jsonify({
            'success': True,
            'flashcards': flashcards
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# START SERVER
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🧪 EIGHTYSIX CHEMISTRY - ULTRA-LIGHTWEIGHT")
    print("=" * 60)
    print(f"🌐 Mode: {'PRODUCTION' if PRODUCTION else 'DEVELOPMENT'}")
    print(f"📚 Books: {len(BOOK_LIBRARY)} available")
    print(f"☁️  Storage: Cloudflare R2")
    print(f"🤖 Search: AI-powered (OpenRouter)")
    print(f"💾 Memory: Ultra-lightweight (no local models)")
    print(f"🔑 API: {'✅ Configured' if OPENROUTER_API_KEY != 'your-key-here' else '❌ Missing'}")
    print(f"📦 R2: {'✅ Configured' if R2_BUCKET_URL != 'https://pub-xxxxx.r2.dev' else '❌ Missing'}")
    print(f"🌐 Port: {PORT}")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=not PRODUCTION)
