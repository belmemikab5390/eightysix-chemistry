"""
EightySix AI-POWERED Server
Now with SMART relevance checking and fallback logic!
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import torch
import requests
import json
from pathlib import Path
import re
import os

app = Flask(__name__)
CORS(app)

# OPENROUTER CONFIGURATION
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', 'your-key-here')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "xiaomi/mimo-v2-flash:free"

# ============================================
# BOOK LIBRARY CONFIGURATION
# ============================================

# BOOK LIBRARY - Currently only Zumdahl is available
BOOK_LIBRARY = {
    'zumdahl': {
        'name': 'General Chemistry',
        'author': 'Zumdahl & Zumdahl',
        'chunks_file': r'data/zumdahl_chunks.json',
        'pdf_file': r'data/zumdahl.pdf'
    },
    'atkins': {
        'name': 'Physical Chemistry',
        'author': 'Atkins & de Paula',
        'chunks_file': r'data/atkins_chunks_with_embeddings.json',
        'pdf_file': r'data/atkins_physical_chemistry.pdf'
    },
    'harris': {
        'name': 'Quantitative Chemical Analysis',
        'author': 'Daniel C. Harris',
        'chunks_file': r'data/harris_chunks_with_embeddings.json',
        'pdf_file': r'data/harris_quantitative_analysis.pdf'
    },
    'klein': {
        'name': 'Organic Chemistry',
        'author': 'David Klein',
        'chunks_file': r'data/klein_chunks_with_embeddings.json',
        'pdf_file': r'data/klein_organic_chemistry.pdf'
    }
}


# Currently loaded book info
current_book_info = {'id': 'zumdahl', 'name': 'General Chemistry', 'author': 'Zumdahl & Zumdahl'}


# ============================================
# SMART SEMANTIC SEARCH WITH RELEVANCE CHECK
# ============================================

class AITextbookSearch:
    # RELEVANCE THRESHOLDS
    HIGH_CONFIDENCE = 0.65   # Strong textbook match
    LOW_CONFIDENCE = 0.35    # Minimum for any relevance
    
    def __init__(self):
        """Initialize with semantic search capabilities"""
        print("🔧 Initializing AI Textbook Search...")
        
        print("📥 Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded!")
        
        self.chunks = self.load_chunks()
        
        if self.chunks:
            print(f"✅ Loaded {len(self.chunks)} chunks with embeddings")
        else:
            print("❌ No chunks loaded!")
        
        self.textbook = {'pages': self.chunks} if self.chunks else None
    
    def load_chunks(self):
        """Load pre-computed chunks with embeddings"""
        chunks_path = Path(r'C:\Users\deffm\Desktop\EightySix\chunks_with_embeddings.json')
        
        if chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            return chunks
        else:
            print(f"⚠️  File not found: {chunks_path}")
            return None
    
    def smart_search(self, question):
        """
        SMART SEARCH with relevance checking
        Returns: (result_dict, is_relevant: bool)
        """
        if not self.chunks:
            return {
                'page': None,
                'text': '',
                'score': 0.0,
                'relevant': False
            }
        
        print(f"\n🔍 Semantic search for: {question}")
        
        try:
            # Encode question
            query_embedding = self.model.encode(question)
            
            # Calculate similarities
            scores = []
            
            for chunk in self.chunks:
                chunk_embedding = torch.tensor(chunk["embedding"], dtype=torch.float32)
                query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
                score = util.cos_sim(query_tensor, chunk_embedding)[0][0].item()
                scores.append((score, chunk))
            
            # Sort by similarity
            scores.sort(reverse=True, key=lambda x: x[0])
            
            # Get best match
            best_score, best_chunk = scores[0]
            
            # Determine confidence level
            if best_score >= self.HIGH_CONFIDENCE:
                confidence = "HIGH"
                is_relevant = True
            elif best_score >= self.LOW_CONFIDENCE:
                confidence = "MEDIUM"
                is_relevant = True
            else:
                confidence = "LOW"
                is_relevant = False
            
            print(f"\n📊 Top 3 matches:")
            for i, (score, chunk) in enumerate(scores[:3], 1):
                text_preview = chunk['text'][:60].replace('\n', ' ')
                print(f"  {i}. Page {chunk['page']} - Score: {score:.3f} - {text_preview}...")
            
            if is_relevant:
                print(f"\n✅ RELEVANT ({confidence})! Using Page {best_chunk['page']} (Score: {best_score:.3f})")
            else:
                print(f"\n⚠️  NOT RELEVANT ({confidence})! Score {best_score:.3f} < {self.LOW_CONFIDENCE}")
                print(f"   Will answer using general knowledge instead.")
            
            return {
                'page': best_chunk['page'],
                'text': best_chunk['text'],
                'score': best_score,
                'relevant': is_relevant,
                'confidence': confidence
            }
        
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'page': None,
                'text': '',
                'score': 0.0,
                'relevant': False
            }
    
    def get_candidate_pages(self, question, top_k=5):
        """Get top K pages with relevance filtering"""
        if not self.chunks:
            return []
        
        query_embedding = self.model.encode(question)
        scores = []
        
        for chunk in self.chunks:
            chunk_embedding = torch.tensor(chunk["embedding"], dtype=torch.float32)
            query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
            score = util.cos_sim(query_tensor, chunk_embedding)[0][0].item()
            
            # Only include if reasonably relevant
            if score >= 0.25:  # Lower threshold for flashcards
                scores.append({
                    'page': chunk['page'],
                    'text': chunk['text'],
                    'score': float(score)
                })
        
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]
    
    def extract_keywords(self, text):
        """Extract keywords (compatibility)"""
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'can', 'you', 'explain', 'tell', 'me', 'about', 'define', 'give'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]

# Add this function after the AITextbookSearch class

def extract_molecules_from_text(text):
    """
    Extract molecule names from AI response.
    Returns list of molecules mentioned.
    """
    # Common chemistry molecules with their formulas
    molecule_patterns = {
        # Simple molecules
        r'\bwater\b': 'water',
        r'\bH2O\b': 'water',
        r'\boxygen\b': 'oxygen',
        r'\bO2\b': 'oxygen',
        r'\bhydrogen\b': 'hydrogen',
        r'\bH2\b': 'hydrogen',
        r'\bcarbon dioxide\b': 'carbon dioxide',
        r'\bCO2\b': 'carbon dioxide',
        r'\bammonia\b': 'ammonia',
        r'\bNH3\b': 'ammonia',
        r'\bmethane\b': 'methane',
        r'\bCH4\b': 'methane',
        r'\bethanol\b': 'ethanol',
        r'\bC2H5OH\b': 'ethanol',
        
        # Organic molecules
        r'\bglucose\b': 'glucose',
        r'\bC6H12O6\b': 'glucose',
        r'\bsucrose\b': 'sucrose',
        r'\bfructose\b': 'fructose',
        r'\bethane\b': 'ethane',
        r'\bpropane\b': 'propane',
        r'\bbutane\b': 'butane',
        r'\bpentane\b': 'pentane',
        r'\bhexane\b': 'hexane',
        r'\bbenzene\b': 'benzene',
        r'\btoluene\b': 'toluene',
        r'\bmethanol\b': 'methanol',
        r'\bacetone\b': 'acetone',
        r'\bpropanol\b': 'propanol',
        
        # Alkanes mentioned in your textbook
        r'\balkane': 'methane',  # Default to simplest alkane
        r'\bethylene\b': 'ethylene',
        r'\bpropylene\b': 'propylene',
        
        # Acids/bases
        r'\bhydrochloric acid\b': 'hydrochloric acid',
        r'\bHCl\b': 'hydrochloric acid',
        r'\bsulfuric acid\b': 'sulfuric acid',
        r'\bH2SO4\b': 'sulfuric acid',
        r'\bacetic acid\b': 'acetic acid',
        r'\bsodium hydroxide\b': 'sodium hydroxide',
        r'\bNaOH\b': 'sodium hydroxide',
        
        # Salts
        r'\bsodium chloride\b': 'sodium chloride',
        r'\bNaCl\b': 'sodium chloride',
        r'\bcalcium carbonate\b': 'calcium carbonate',
        r'\bCaCO3\b': 'calcium carbonate',
    }
    
    detected = []
    text_lower = text.lower()
    
    for pattern, molecule in molecule_patterns.items():
        if re.search(pattern, text_lower):
            if molecule not in detected:
                detected.append(molecule)
    
    return detected[:3]  # Return max 3 molecules to avoid clutter

# Initialize AI search
ai_search = AITextbookSearch()


# ============================================
# FLASK ROUTES WITH SMART PROMPTS
# ============================================
def get_complexity_instruction(level):
    """
    Return complexity-specific instructions for the AI
    Level 1-10 scale
    """
    
    instructions = {
        1: """
COMPLEXITY LEVEL 1 - Explain Like I'm 5 Years Old:
- Use VERY simple language (no jargon)
- Use everyday analogies (cookies, toys, sports)
- Short sentences (under 15 words each)
- No complex chemical formulas
- Make it FUN and engaging
- Example: "Atoms are like tiny Lego blocks that stick together!"
""",
        2: """
COMPLEXITY LEVEL 2 - Elementary School:
- Simple language with basic science terms
- Relatable analogies (school, home, playground)
- Introduce simple concepts gently
- Use basic formulas: H₂O (write it out: "H-2-O")
- Encouraging tone
- Example: "Water is made of 2 hydrogen atoms and 1 oxygen atom stuck together!"
""",
        3: """
COMPLEXITY LEVEL 3 - Middle School:
- Introduce proper chemistry terms but explain them
- Use school-appropriate analogies
- Basic equations with explanations
- Simple LaTeX: $H_2O$ but always explain what it means
- Build on prior knowledge
- Example: "Water ($H_2O$) forms when hydrogen and oxygen chemically bond together."
""",
        4: """
COMPLEXITY LEVEL 4 - Early High School:
- Use chemistry vocabulary (define new terms)
- Basic stoichiometry concepts
- Simple balanced equations
- LaTeX for formulas: $\\ce{H2O}$, $PV = nRT$
- Connect to real-world applications
- Example: "In the reaction $\\ce{2H2 + O2 -> 2H2O}$, hydrogen and oxygen combine to form water."
""",
        5: """
COMPLEXITY LEVEL 5 - High School Chemistry:
- Standard chemistry terminology
- Balanced equations expected
- Full LaTeX notation: $\\ce{2H2 + O2 -> 2H2O}$
- Stoichiometry, molarity, basic thermodynamics
- Explain mechanisms when relevant
- Example: "The combustion reaction $\\ce{2H2 + O2 -> 2H2O}$ releases 483.6 kJ/mol."
""",
        6: """
COMPLEXITY LEVEL 6 - Advanced High School / AP Chemistry:
- Advanced concepts (thermodynamics, kinetics, equilibrium)
- Detailed mechanisms
- Multiple equations: $K_a$, $K_b$, $K_{sp}$
- Expect understanding of: enthalpy, entropy, Gibbs free energy
- Mathematical derivations when helpful
- Example: "At equilibrium, $\\Delta G = 0$, so $\\Delta H - T\\Delta S = 0$, giving us $T = \\frac{\\Delta H}{\\Delta S}$"
""",
        7: """
COMPLEXITY LEVEL 7 - College / Undergraduate:
- Rigorous chemistry concepts
- Quantum mechanics basics (orbitals, electron configurations)
- Detailed reaction mechanisms
- Multiple equilibria, Le Chatelier's principle applications
- Integration of concepts across topics
- Example: "The reaction coordinate diagram shows the transition state at the activation energy maximum, where $\\Delta G^‡$ determines the rate constant via the Eyring equation."
""",
        8: """
COMPLEXITY LEVEL 8 - Advanced Undergraduate:
- Advanced physical chemistry
- Quantum mechanical descriptions
- Statistical mechanics basics
- Detailed orbital theory
- Research paper level language
- Example: "The HOMO-LUMO gap determines the electronic transition energy. For benzene, the π → π* transition occurs at 254 nm due to the molecular orbital splitting."
""",
        9: """
COMPLEXITY LEVEL 9 - Graduate Level:
- Graduate-level theory
- Advanced quantum chemistry
- Spectroscopy, crystallography details
- Computational chemistry concepts
- Literature-quality explanations
- Example: "Using time-dependent density functional theory (TD-DFT) with the B3LYP functional and 6-311+G(d,p) basis set..."
""",
        10: """
COMPLEXITY LEVEL 10 - PhD / Research Level:
- Cutting-edge chemistry concepts
- Advanced mathematical derivations
- Quantum field theory applications if relevant
- Latest research methodologies
- Assume expert background knowledge
- Example: "The non-adiabatic coupling matrix elements were calculated using complete active space self-consistent field (CASSCF) methods, with dynamic correlation via multi-reference perturbation theory (MRPT2)..."
"""
    }
    
    # Return closest level
    closest_level = min(instructions.keys(), key=lambda x: abs(x - level))
    return instructions[closest_level]

@app.route('/chat', methods=['POST'])
def chat():
    """AI chat with SMART relevance checking"""
    try:
        data = request.json
        question = data.get('message', '')
        mode = data.get('mode', 'study')
        complexity = data.get('complexity', 5)

        print(f"\n💬 Question: {question}")
        print(f"🎯 Mode: {mode}")
        print(f"📊 Complexity: {complexity}/10")

        # SMART SEARCH with relevance check
        result = ai_search.smart_search(question)
        
        page = result['page']
        textbook_content = result['text']
        score = result['score']
        is_relevant = result['relevant']

        complexity_instruction = get_complexity_instruction(complexity)
        print(f"📝 Using instruction: {complexity_instruction[:200]}")

        # BUILD SMART PROMPTS based on relevance
        
        if is_relevant:
            # ✅ TEXTBOOK CONTENT IS RELEVANT - Use it!
            print(f"📖 Using textbook content from Page {page}")
            
            if mode == 'study':
                # ← ADD complexity_instruction to your system prompt
                system_prompt = f"""You are a chemistry tutor helping students learn from their textbook.

{complexity_instruction}

IMPORTANT RULES:
- Explain concepts clearly using the provided textbook content
- Use LaTeX for ALL formulas: $H_2O$, $PV = nRT$, etc.
- Use proper chemical notation: $\\ce{{H2O}}$, $\\ce{{CO2}}$
- Be accurate and cite the textbook when appropriate
- MATCH THE COMPLEXITY LEVEL SPECIFIED ABOVE

FORMAT:
📖 Concept:
[Clear explanation with LaTeX at appropriate complexity level]

💡 Example:
[Real-world example matching complexity level]

✨ Key Points:
- [Important points with LaTeX]"""
            
            elif mode == 'exam':
                system_prompt = f"""You are creating exam questions from textbook content.

{complexity_instruction}

IMPORTANT RULES:
- Base questions on the provided textbook content
- Use LaTeX for all chemistry and math
- Make questions challenging but fair at the specified complexity level
- Provide clear explanations

FORMAT:
Question: [Question with LaTeX at appropriate level]

A) [Option]
B) [Option]
C) [Option]
D) [Option]

Correct Answer: [Letter] ✅

Explanation: [Why this is correct with LaTeX]"""
            
            elif mode == 'practice':
                system_prompt = f"""You are a problem-solving tutor using textbook content.

{complexity_instruction}

IMPORTANT RULES:
- Create practice problems based on textbook content
- Show ALL steps with LaTeX
- Use proper chemical notation
- Teach the method clearly at the specified complexity level

FORMAT:
🎯 Problem: [Problem statement]

📋 Given:
- [Values with LaTeX]

💡 Solution:
Step 1: [Description]
$$calculation$$

✅ Final Answer: [Answer with LaTeX]"""
            
            elif mode == 'summary':
                system_prompt = f"""Create a study summary from textbook content.

{complexity_instruction}

IMPORTANT RULES:
- Summarize the provided textbook content
- Use LaTeX for all formulas
- Be concise but complete
- Organize information clearly at the specified complexity level

FORMAT:
📚 Summary: {{question}}
📖 Source: Page [X]

📌 Key Concepts:
- [Concept with LaTeX]

📐 Important Formulas:
- $formula$ — [explanation]

⚡ Quick Facts:
- [Facts with LaTeX]"""
            
            else:
                system_prompt = f"""You are a helpful chemistry tutor.

{complexity_instruction}

Use the textbook content to answer accurately. Use LaTeX for formulas: $formula$
Match the complexity level specified above."""
            
            user_message = f"""TEXTBOOK CONTENT (Page {page}):
{textbook_content[:1500]}

STUDENT'S QUESTION: {question}

COMPLEXITY LEVEL: {complexity}/10

Use the textbook content above to answer the question at the specified complexity level. Use LaTeX notation for ALL formulas and equations."""

        else:
            # ⚠️ TEXTBOOK CONTENT NOT RELEVANT - Answer like Normal GPT!
            print(f"💭 Using general knowledge (score: {score:.3f})")
            
            if mode == 'study':
                system_prompt = f"""You are a helpful chemistry tutor with general knowledge.

{complexity_instruction}

IMPORTANT INSTRUCTION:
The student's question is not covered in their specific textbook.
Answer naturally using your general chemistry knowledge at the specified complexity level.

RULES:
- Answer naturally and helpfully at complexity level {complexity}/10
- Be accurate and educational
- Use LaTeX for formulas: $H_2O$, $PV = nRT$
- At the END, add: "💡 Note: This is general chemistry knowledge, not from your specific textbook."

Answer the question directly and naturally."""
            
            elif mode == 'exam':
                system_prompt = f"""You are a helpful chemistry tutor.

{complexity_instruction}

IMPORTANT INSTRUCTION:
This topic is not in the student's textbook.

RESPONSE:
"⚠️ This topic isn't covered in your chemistry textbook.

However, I can help you with:
- General chemistry questions (I'll answer using general knowledge at level {complexity}/10)
- Topics that ARE in your textbook (stoichiometry, acids/bases, thermodynamics, etc.)

What would you like to know about?"
"""
            
            elif mode == 'practice':
                system_prompt = f"""You are a helpful chemistry tutor.

{complexity_instruction}

IMPORTANT INSTRUCTION:
This topic is not in the student's textbook.

If chemistry-related: Provide a general practice problem at complexity level {complexity}/10 with this note: "💡 General chemistry problem - not from your textbook"
If not chemistry: Suggest focusing on textbook topics instead.

Be helpful but honest about the source."""
            
            elif mode == 'summary':
                system_prompt = f"""You are a helpful chemistry tutor.

{complexity_instruction}

IMPORTANT INSTRUCTION:
This topic is not in the student's textbook.

RESPONSE:
"⚠️ This topic isn't found in your chemistry textbook.

Your textbook covers:
- Stoichiometry
- Thermodynamics  
- Acids & Bases
- Chemical Bonding
- Organic Chemistry
Would you like a summary of a topic that IS in your textbook instead?"
"""
            
            else:
                system_prompt = f"""You are a helpful chemistry tutor with general knowledge.

{complexity_instruction}

The student's question is not in their textbook.
Answer naturally using your general chemistry knowledge at complexity level {complexity}/10.
Be helpful, accurate, and educational.

At the end, add: "💡 General knowledge - not from your textbook."""
            
            user_message = f"""STUDENT'S QUESTION: {question}

COMPLEXITY LEVEL: {complexity}/10

CONTEXT:
This question is not covered in the student's chemistry textbook.
Answer naturally using your general chemistry knowledge at the specified complexity level.
Be helpful and educational."""
        
        print("🤖 Calling OpenRouter API...")
        
        # Call OpenRouter API
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "EightySix"
        }
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result_json = response.json()
            ai_response = result_json['choices'][0]['message']['content']
            
            print("✅ Response complete!")

            detected_molecules = extract_molecules_from_text(ai_response)

            if detected_molecules:
                print(f"🧪 Detected molecules: {detected_molecules}")
            return jsonify({
                'success': True,
                'response': ai_response,
                'molecules': detected_molecules,  # ← ADDED THIS LINE
                'complexity_used': complexity,
                'source': {
                    'textbook': 'Zumdahl General Chemistry',
                    'page': page if is_relevant else None,
                    'chapter': f'Page {page}' if is_relevant else 'Not in textbook',
                    'mode': mode,
                    'relevance_score': score,
                    'used_textbook': is_relevant
                }
            })
        else:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', 'Unknown error')
            print(f"❌ OpenRouter error: {error_msg}")
            raise Exception(f"OpenRouter error: {error_msg}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/generate-flashcards', methods=['POST'])
def generate_flashcards():
    """Generate flashcards with relevance checking"""
    try:
        data = request.json
        topic = data.get('topic', '')
        count = data.get('count', 5)
        
        print(f"\n🎴 Generating {count} flashcards for: {topic}")
        
        # Get relevant pages
        pages = ai_search.get_candidate_pages(topic, top_k=3)
        
        if not pages or (pages and pages[0]['score'] < 0.3):
            return jsonify({
                'success': False,
                'error': f'Topic "{topic}" not found in textbook. Try topics like: molarity, pH, thermodynamics, bonding, stoichiometry.'
            }), 404
        
        # Combine content
        combined_content = "\n\n".join([p['text'][:1000] for p in pages])
        
        user_message = f"""Create {count} study flashcards about {topic} from this textbook content:

{combined_content}

RESPOND ONLY IN THIS JSON FORMAT (no other text):
{{
  "flashcards": [
    {{"front": "What is X?", "back": "Clear answer"}},
    {{"front": "Define Y", "back": "Concise definition"}}
  ]
}}

Create exactly {count} flashcards."""

        print("🤖 Calling OpenRouter API...")
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "EightySix"
        }
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You create study flashcards. Always respond with valid JSON only."},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }
        
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result['choices'][0]['message']['content']
            
            # Extract JSON
            try:
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    flashcard_data = json.loads(json_match.group())
                    flashcards = flashcard_data.get('flashcards', [])
                    flashcards = flashcards[:count]
                    
                    print(f"✅ Generated {len(flashcards)} flashcards!")
                    
                    return jsonify({
                        'success': True,
                        'flashcards': flashcards,
                        'pages': [p['page'] for p in pages]
                    })
            except Exception as e:
                print(f"⚠️ JSON parse error: {e}")
        
        return jsonify({
            'success': False,
            'error': 'Could not generate flashcards'
        }), 500
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@app.route('/load-book', methods=['POST'])
def load_book():
    """Load a book from the library"""
    global ai_search, current_book_info
    
    try:
        data = request.json
        book_id = data.get('bookId')
        
        print(f"\n📚 Request to load book: {book_id}")
        
        if book_id not in BOOK_LIBRARY:
            return jsonify({
                'success': False,
                'error': f'Book "{book_id}" not found in library. Currently only "zumdahl" is available.'
            }), 404
        
        book = BOOK_LIBRARY[book_id]
        chunks_path = Path(book['chunks_file'])
        
        if not chunks_path.exists():
            return jsonify({
                'success': False,
                'error': f'Chunks file not found at: {chunks_path}'
            }), 404
        
        print(f"📖 Loading: {book['name']} by {book['author']}")
        print(f"📂 From: {chunks_path}")
        
        # Load the chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Update the AI search system with new chunks
        ai_search.chunks = chunks
        ai_search.textbook = {'pages': chunks}
        
        # Update current book info
        current_book_info = {
            'id': book_id,
            'name': book['name'],
            'author': book['author']
        }
        
        print(f"✅ Successfully loaded {len(chunks)} chunks from {book['name']}")
        
        return jsonify({
            'success': True,
            'book_id': book_id,
            'book_name': book['name'],
            'author': book['author'],
            'chunks_count': len(chunks)
        })
        
    except Exception as e:
        print(f"❌ Error loading book: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/get-library', methods=['GET'])
def get_library():
    """Get list of available books in the library"""
    books = []
    for book_id, book_info in BOOK_LIBRARY.items():
        chunks_path = Path(book_info['chunks_file'])
        books.append({
            'id': book_id,
            'name': book_info['name'],
            'author': book_info['author'],
            'available': chunks_path.exists()
        })
    
    return jsonify({
        'success': True,
        'books': books,
        'current_book': current_book_info
    })


@app.route('/pdf/<book_id>', methods=['GET'])
def serve_pdf(book_id):
    """Serve PDF file for a book"""
    try:
        if book_id not in BOOK_LIBRARY:
            return jsonify({
                'success': False,
                'error': f'Book "{book_id}" not found'
            }), 404
        
        book = BOOK_LIBRARY[book_id]
        
        # Check if book has a PDF file
        if 'pdf_file' not in book:
            return jsonify({
                'success': False,
                'error': 'No PDF file configured for this book'
            }), 404
        
        pdf_path = Path(book['pdf_file'])
        
        if not pdf_path.exists():
            return jsonify({
                'success': False,
                'error': f'PDF file not found: {pdf_path}'
            }), 404
        
        print(f"📄 Serving PDF: {book['name']} from {pdf_path}")
        
        # Serve the PDF file
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=False,
            download_name=f"{book_id}.pdf"
        )
        
    except Exception as e:
        print(f"❌ Error serving PDF: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'textbook_loaded': ai_search.textbook is not None,
        'ai_powered': True,
        'using': 'OpenRouter + Smart Semantic Search',
        'high_confidence': ai_search.HIGH_CONFIDENCE,
        'low_confidence': ai_search.LOW_CONFIDENCE
    })


# ============================================
# START SERVER
# ============================================


if __name__ == '__main__':
    print("=" * 60)
    print("🤖 EIGHTYSIX AI-POWERED SERVER WITH LIBRARY")
    print("Using Smart Semantic Search + Relevance Checking")
    print("=" * 60)
    
    if ai_search.chunks:
        print(f"✅ Default Textbook: {current_book_info['name']}")
        print(f"   by {current_book_info['author']}")
        print(f"📚 Loaded: {len(ai_search.chunks)} chunks")
        print(f"🧠 SMART SEARCH ACTIVE!")
        print(f"⚡ High confidence: {ai_search.HIGH_CONFIDENCE}")
        print(f"⚡ Low confidence: {ai_search.LOW_CONFIDENCE}")
        print(f"🎴 Flashcard Generator Ready!")
    else:
        print("❌ No textbook loaded!")
    
    print(f"\n📖 Available Books in Library:")
    for book_id, book_info in BOOK_LIBRARY.items():
        status = "✅" if Path(book_info['chunks_file']).exists() else "❌"
        print(f"  {status} {book_info['name']} ({book_id})")
    
    print(f"\n✅ OpenRouter API: Configured")
    print("🌐 Server: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
