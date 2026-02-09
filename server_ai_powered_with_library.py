"""
EightySix AI-POWERED Server - Production Version
Configured for Render.com deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import requests
import json
import os

app = Flask(__name__)
CORS(app)

# OPENROUTER CONFIGURATION
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', 'your-key-here')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "stepfun/step-3.5-flash:free"

# ============================================
# AI TEXTBOOK SEARCH
# ============================================

class AITextbookSearch:
    HIGH_CONFIDENCE = 0.65
    LOW_CONFIDENCE = 0.35
    
    def __init__(self):
        print("🔧 Initializing AI Textbook Search...")
        print("📥 Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded!")
        
        self.chunks = self.create_sample_chunks()
        print(f"✅ Loaded {len(self.chunks)} sample chunks")
        self.textbook = {'pages': self.chunks}
    
    def create_sample_chunks(self):
        """Create sample chemistry data"""
        sample_texts = [
            "Chemistry is the study of matter and its properties. Matter is anything that has mass and takes up space.",
            "An atom consists of a nucleus containing protons and neutrons, surrounded by electrons.",
            "The periodic table organizes elements by atomic number and electron configuration.",
            "Chemical bonds form when atoms share or transfer electrons. Types include ionic, covalent, and metallic bonds.",
            "Chemical reactions involve breaking and forming of chemical bonds. Reactants transform into products.",
            "The mole is a unit for counting particles. One mole contains 6.022 × 10²³ particles (Avogadro's number).",
            "Stoichiometry calculates quantities in chemical reactions based on balanced equations.",
            "Acids donate protons (H⁺) while bases accept protons. The pH scale measures acidity from 0 to 14.",
            "Thermodynamics studies energy changes in reactions. Enthalpy measures heat content.",
            "Reaction rates depend on temperature, concentration, surface area, and catalysts."
        ]
        
        chunks = []
        for i, text in enumerate(sample_texts, 1):
            embedding = self.model.encode(text)
            chunks.append({
                'page': i,
                'text': text,
                'embedding': embedding.tolist()
            })
        
        return chunks
    
    def smart_search(self, question):
        """Search with relevance checking"""
        if not self.chunks:
            return "No data available.", 0.0, False
        
        question_embedding = self.model.encode(question)
        
        best_chunks = []
        for chunk in self.chunks:
            similarity = util.cos_sim(question_embedding, chunk['embedding']).item()
            best_chunks.append((chunk, similarity))
        
        best_chunks.sort(key=lambda x: x[1], reverse=True)
        top_chunk, top_similarity = best_chunks[0]
        
        is_relevant = top_similarity >= self.LOW_CONFIDENCE
        context_chunks = best_chunks[:3]
        context = "\n\n".join([
            f"[Page {chunk['page']}] {chunk['text']}"
            for chunk, _ in context_chunks
        ])
        
        return context, top_similarity, is_relevant


ai_search = AITextbookSearch()


# ============================================
# AI RESPONSE
# ============================================

def get_ai_response(question, context, complexity=3, is_relevant=True):
    """Get AI response via OpenRouter"""
    
    if not is_relevant:
        prompt = f"""You are a helpful chemistry tutor.

The user asked: "{question}"

This doesn't appear to be about chemistry. Politely redirect them to chemistry topics."""
    else:
        complexity_levels = {
            1: "Explain to a beginner using simple language.",
            2: "Explain like to a high school student.",
            3: "Balanced explanation with proper terminology.",
            4: "Detailed explanation for advanced students.",
            5: "Comprehensive university-level explanation."
        }
        
        complexity_instruction = complexity_levels.get(complexity, complexity_levels[3])
        
        prompt = f"""You are a chemistry tutor.

TEXTBOOK CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
{complexity_instruction}

Provide a clear answer based on the context."""
    
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
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

@app.route('/ask', methods=['POST'])
def ask():
    """Ask questions"""
    try:
        data = request.json
        question = data.get('question', '')
        complexity = data.get('complexity', 3)
        
        print(f"\n📝 Question: {question}")
        
        context, similarity, is_relevant = ai_search.smart_search(question)
        print(f"🔍 Similarity: {similarity:.3f}, Relevant: {is_relevant}")
        
        answer = get_ai_response(question, context, complexity, is_relevant)
        
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
        
        prompt = f"""Create {count} flashcards about {topic} in chemistry.

Format:
Q: [Question]
A: [Answer]"""
        
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            text = response.json()['choices'][0]['message']['content']
            
            flashcards = []
            lines = text.strip().split('\n')
            current_q = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Q:'):
                    current_q = line[2:].strip()
                elif line.startswith('A:') and current_q:
                    flashcards.append({
                        'question': current_q,
                        'answer': line[2:].strip()
                    })
                    current_q = None
            
            return jsonify({'success': True, 'flashcards': flashcards})
        else:
            return jsonify({'success': False, 'error': f'API error'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'mode': 'demo',
        'chunks_count': len(ai_search.chunks),
        'ai_powered': True
    })


@app.route('/get-library', methods=['GET'])
def get_library():
    """Get books"""
    return jsonify({
        'success': True,
        'books': [{
            'id': 'demo',
            'name': 'Chemistry Demo',
            'author': 'Sample',
            'available': True
        }],
        'current_book': {'id': 'demo', 'name': 'Chemistry Demo'}
    })


# ============================================
# START SERVER
# ============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("=" * 60)
    print("🤖 EIGHTYSIX AI SERVER - DEMO MODE")
    print("=" * 60)
    print(f"📚 Chunks: {len(ai_search.chunks)}")
    print(f"✅ OpenRouter: {'OK' if OPENROUTER_API_KEY != 'your-key-here' else 'Not set'}")
    print(f"🌐 Port: {port}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)
