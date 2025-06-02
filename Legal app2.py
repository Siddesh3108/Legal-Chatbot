import os
import logging
import numpy as np
import torch
import uuid
from flask import Flask, request, jsonify, send_from_directory, render_template
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import transformers
import fitz
import re
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress unnecessary Hugging Face warnings
transformers.logging.set_verbosity_error()

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
PDF_FOLDER = os.path.join(UPLOAD_FOLDER, 'pdfs')
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(PDF_FOLDER, exist_ok=True)

# Load ML models (with error handling)
try:
    # Load QA model for precise answer extraction
    qa_model_name = "deepset/roberta-base-squad2"
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)
    
    # Load sentence transformer for semantic similarity
    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Load summarization pipeline for long documents
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    logger.info("ML models loaded successfully")
except Exception as e:
    logger.error(f"Error loading ML models: {str(e)}")
    # Fall back to simpler models if primary ones fail
    try:
        qa_pipeline = pipeline('question-answering')
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Fell back to simpler ML models")
    except Exception as e:
        logger.error(f"Error loading fallback models: {str(e)}")
        sentence_model = None
        qa_pipeline = None

def extract_text_from_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            page_text = page.get_text("text") # Get plain text
            if page_text:
                # Basic cleaning - might need more sophisticated cleaning
                page_text = re.sub(r'\s+', ' ', page_text).strip()
                text += page_text + "\n\n"
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting text from PDF using PyMuPDF: {str(e)}")
        return "" # Fallback or raise error
    return text

# Enhanced legal database with broader coverage
legal_database = {
    "leave": {
        "text": "As per labor laws, employees are entitled to various types of leave including casual leave, sick leave, and earned leave. Typically, employees receive 12-15 casual leaves, 7-10 sick leaves, and 15-30 earned leaves annually, depending on company policy and applicable state laws.",
        "keywords": ["leave", "vacation", "time off", "holiday", "absent"],
        "references": ["Factories Act, 1948", "Shops and Establishments Act", "State-specific labor laws"]
    },
    "bail": {
        "text": "Under the Bharatiya Nagarik Suraksha Sanhita, 2023, bail provisions have been modernized. For non-bailable offenses with punishment less than 7 years, notice before arrest is mandatory. The court may grant bail based on factors including severity of offense, likelihood of flight risk, and chances of evidence tampering.",
        "keywords": ["bail", "arrest", "bailable", "jail", "release"],
        "references": ["Bharatiya Nagarik Suraksha Sanhita, 2023", "Code of Criminal Procedure"]
    },
    "evidence": {
        "text": "The Bharatiya Sakshya Adhiniyam governs evidence admissibility in Indian courts. Electronic evidence is now explicitly recognized, requiring proper chain of custody and authentication. Hearsay evidence is generally inadmissible with specific exceptions such as dying declarations.",
        "keywords": ["evidence", "proof", "witness", "testimony", "document"],
        "references": ["Bharatiya Sakshya Adhiniyam", "Indian Evidence Act"]
    },
    "arrest": {
        "text": "Police must inform the arrested person about the grounds of arrest, right to bail, and right to legal representation. A medical examination is mandatory after arrest. The arrested person must be presented before a magistrate within 24 hours of arrest.",
        "keywords": ["arrest", "police", "detention", "custody", "rights"],
        "references": ["Code of Criminal Procedure", "Constitution of India Article 22"]
    },
    "property": {
        "text": "Property laws in India are governed by various statutes including the Transfer of Property Act. For immovable property, registration is mandatory for sales. Ancestral property follows succession laws based on religion, while self-acquired property can be disposed of through a will.",
        "keywords": ["property", "real estate", "land", "house", "ownership"],
        "references": ["Transfer of Property Act", "Indian Succession Act", "Hindu Succession Act"]
    },
    "divorce": {
        "text": "Divorce can be obtained through mutual consent after a 6-month cooling period, or through contested means on grounds such as cruelty, desertion, and adultery. The process typically takes 6-12 months for mutual consent and 2-5 years for contested divorce.",
        "keywords": ["divorce", "marriage", "separation", "alimony", "maintenance"],
        "references": ["Hindu Marriage Act", "Special Marriage Act"]
    },
    "contract": {
        "text": "A valid contract requires offer, acceptance, consideration, capacity to contract, and lawful object. Breaches can be remedied through specific performance or damages. Force majeure clauses excuse non-performance due to unforeseeable circumstances.",
        "keywords": ["contract", "agreement", "breach", "enforce", "terms"],
        "references": ["Indian Contract Act", "Specific Relief Act"]
    },
    "criminal": {
        "text": "Criminal offenses in India are categorized as cognizable or non-cognizable, and bailable or non-bailable. Serious offenses like murder, rape, and rash driving causing death are cognizable and non-bailable. For rash driving causing death (Section 304A), the punishment can be imprisonment up to 2 years or fine or both.",
        "keywords": ["criminal", "offense", "crime", "murder", "theft", "rash", "negligent", "driving", "accident", "kill", "death"],
        "references": ["Indian Penal Code", "Bharatiya Nyaya Sanhita, 2023", "Motor Vehicles Act"]
    },
    "traffic": {
        "text": "Under the Motor Vehicles Act, rash and negligent driving is a punishable offense. When such driving results in death, it's considered culpable homicide not amounting to murder under Section 304A of IPC, punishable with imprisonment up to 2 years or fine or both. For accidents in school zones or other protected areas, penalties are typically enhanced.",
        "keywords": ["traffic", "driving", "vehicle", "accident", "speed", "rash", "negligent", "license", "school zone", "km/hr"],
        "references": ["Motor Vehicles Act", "Indian Penal Code Section 304A", "Local Traffic Regulations"]
    },
    "family": {
        "text": "Family law in India covers marriage, divorce, adoption, maintenance, and child custody. Different personal laws apply based on religion, though Special Marriage Act provides for civil marriages. Child custody decisions prioritize the welfare of the child above parental rights.",
        "keywords": ["family", "child", "custody", "adoption", "guardian", "succession"],
        "references": ["Hindu Marriage Act", "Special Marriage Act", "Juvenile Justice Act", "Guardian and Wards Act"]
    },
    "employment": {
        "text": "Employment laws in India govern working conditions, minimum wages, benefits, and termination procedures. The employer-employee relationship is regulated through various central and state laws, with specific provisions for different industries.",
        "keywords": ["job", "work", "employee", "salary", "termination", "workplace", "labor"],
        "references": ["Industrial Disputes Act", "Factories Act", "Employees Provident Fund Act"]
    }
}

# Import and process the specified PDF files
pdf_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sakshya.pdf"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "bharatiya-sakshya-bill-511326.pdf"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Laws.pdf")
]

for pdf_path in pdf_paths:
    try:
        if os.path.exists(pdf_path):
            filename = os.path.basename(pdf_path)
            doc_id = os.path.splitext(filename)[0].lower().replace(" ", "_").replace(",", "")
            
            pdf_text = extract_text_from_pdf(pdf_path)
            logger.info(f"Processing PDF: {filename}, Extracted {len(pdf_text)} characters")
            
            if pdf_text:
                # Add the PDF content to the legal database
                legal_database[doc_id] = {
                    "text": pdf_text[:5000] + ("..." if len(pdf_text) > 5000 else ""),
                    "keywords": [doc_id.split("_")[0], doc_id.replace("_", " ")],
                    "references": [filename],
                    "full_text": pdf_text
                }
                logger.info(f"Successfully added to legal database: {filename}")
            else:
                logger.warning(f"Could not extract text from PDF: {filename}")
        else:
            logger.warning(f"PDF file not found: {pdf_path}")
    except Exception as e:
        logger.error(f"Error importing PDF {pdf_path}: {str(e)}")

# Store custom PDF data with enhanced metadata
custom_pdf_data = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def semantic_search(query, documents, threshold=0.4):  # Lower threshold for better recall
    """Perform semantic search using sentence transformers."""
    if not sentence_model or not documents:
        return []
    
    try:
        query_embedding = sentence_model.encode(query, convert_to_tensor=True)
        doc_embeddings = sentence_model.encode(documents, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
        results = [(score.item(), doc) for score, doc in zip(cos_scores, documents) if score > threshold]
        results.sort(reverse=True, key=lambda x: x[0])
        return results
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return []

def extract_answer_from_context(question, context):
    """Use QA model to extract precise answer from context."""
    if not qa_pipeline or not context.strip():
        return None
    
    try:
        result = qa_pipeline({
            'question': question,
            'context': context
        })
        return result['answer'] if result['score'] > 0.1 else None
    except Exception as e:
        logger.error(f"Error in QA extraction: {str(e)}")
        return None

def summarize_text(text, max_length=150):
    """Summarize long text using BART model."""
    if not summarizer or not text.strip():
        return text
    
    try:
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        return " ".join(summaries)
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        return text[:max_length] + "..." if len(text) > max_length else text

def search_pdf_content(query, pdf_text):
    """Search for relevant content in PDF text using ML-enhanced methods."""
    if not pdf_text.strip():
        return None
    
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', pdf_text) if p.strip()]
    semantic_results = semantic_search(query, paragraphs)
    
    if semantic_results:
        best_match = semantic_results[0][1]
        precise_answer = extract_answer_from_context(query, best_match)
        if precise_answer:
            return precise_answer
        return best_match
    
    query_words = set(query.lower().split())
    relevant_paragraphs = []
    for para in paragraphs:
        para_lower = para.lower()
        if any(word in para_lower for word in query_words):
            relevant_paragraphs.append(para)
    
    return relevant_paragraphs[0] if relevant_paragraphs else None

def extract_entities(text):
    """Extract meaningful legal entities from text for better matching."""
    # Simple entity extraction - could be enhanced with NER models
    entities = []
    
    # Extract legal terms 
    legal_terms = ["murder", "theft", "assault", "divorce", "property", "contract", 
                   "driving", "accident", "evidence", "bail", "arrest", "crime",
                   "negligent", "rash", "killed", "death", "injured", "damage", 
                   "compensation", "school", "zone", "speed", "km/hr"]
    
    words = text.lower().split()
    for term in legal_terms:
        if term in words or any(term in word for word in words):
            entities.append(term)
    
    return entities

def find_legal_reference(question):
    """Find the most relevant legal reference using ML with a lowered threshold."""
    # Extract entities to improve matching
    entities = extract_entities(question)
    
    # First try direct keyword matching
    for topic, data in legal_database.items():
        if any(keyword in question.lower() for keyword in data['keywords']):
            return data['text'], data['references'], 0.9
        
        # Also check if any extracted entities match keywords
        if entities and any(entity in data['keywords'] for entity in entities):
            return data['text'], data['references'], 0.8
    
    # If no direct match, try semantic matching
    if sentence_model:
        try:
            question_embedding = sentence_model.encode(question, convert_to_tensor=True)
            best_score = 0
            best_topic = None
            
            for topic, data in legal_database.items():
                topic_text = f"{topic}. {' '.join(data['keywords'])}. {data['text']}"
                topic_embedding = sentence_model.encode(topic_text, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(question_embedding, topic_embedding).item()
                
                if similarity > best_score:
                    best_score = similarity
                    best_topic = topic
            
            # Lower threshold for better recall but track confidence
            if best_score > 0.3 and best_topic:
                return legal_database[best_topic]['text'], legal_database[best_topic]['references'], best_score
        except Exception as e:
            logger.error(f"Error in semantic legal reference matching: {str(e)}")
    
    # Fallback for specific case types
    if any(term in question.lower() for term in ["driving", "accident", "kill", "death", "school", "zone", "speed"]):
        return legal_database["traffic"]["text"], legal_database["traffic"]["references"], 0.7
    
    return None, None, 0


os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    try:
        if 'pdfFile' not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400
            
        file = request.files['pdfFile']
        document_type = request.form.get('documentType', 'other')
        document_description = request.form.get('documentDescription', '')
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
            
        if file and allowed_file(file.filename):
            file_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            file_path = os.path.join(PDF_FOLDER, f"{file_id}_{filename}")
            file.save(file_path)
            
            pdf_text = extract_text_from_pdf(file_path)
            if not pdf_text.strip():
                os.remove(file_path)
                return jsonify({"success": False, "error": "Could not extract text from PDF"}), 400
            
            custom_pdf_data[file_id] = {
                "id": file_id,
                "filename": filename,
                "path": file_path,
                "type": document_type,
                "description": document_description,
                "text": pdf_text,
                "processed": False,
                "embeddings": None
            }
            
            try:
                if sentence_model:
                    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', pdf_text) if p.strip()]
                    custom_pdf_data[file_id]['embeddings'] = sentence_model.encode(paragraphs)
                    custom_pdf_data[file_id]['paragraphs'] = paragraphs
                    custom_pdf_data[file_id]['processed'] = True
            except Exception as e:
                logger.error(f"Error processing PDF embeddings: {str(e)}")
                custom_pdf_data[file_id]['processed'] = False
            
            logger.info(f"Uploaded PDF: {filename}, ID: {file_id}")
            return jsonify({
                "success": True, 
                "message": "PDF uploaded successfully",
                "id": file_id
            })
        else:
            return jsonify({"success": False, "error": "Only PDF files are allowed"}), 400
            
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": "An error occurred while processing your file"}), 500

@app.route('/pdf-list', methods=['GET'])
def pdf_list():
    try:
        pdfs = []
        for pdf_id, pdf_info in custom_pdf_data.items():
            pdfs.append({
                "id": pdf_id,
                "filename": pdf_info["filename"],
                "type": pdf_info["type"],
                "description": pdf_info["description"],
                "processed": pdf_info.get("processed", False)
            })
        return jsonify({"success": True, "pdfs": pdfs})
    except Exception as e:
        logger.error(f"Error getting PDF list: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": "An error occurred while retrieving PDF list"}), 500

@app.route('/delete-pdf/<pdf_id>', methods=['DELETE'])
def delete_pdf(pdf_id):
    try:
        if pdf_id in custom_pdf_data:
            try:
                os.remove(custom_pdf_data[pdf_id]["path"])
            except OSError:
                pass
            del custom_pdf_data[pdf_id]
            return jsonify({"success": True, "message": "PDF deleted successfully"})
        else:
            return jsonify({"success": False, "error": "PDF not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting PDF: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": "An error occurred while deleting the PDF"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Invalid request. Please provide a 'question' field in JSON format."}), 400

        user_question = data["question"].strip()
        logger.info(f"Received question: {user_question}")

        if not user_question:
            return jsonify({"answer": "Please provide a valid legal question."})

        # Extract entities for better matching
        entities = extract_entities(user_question)
        logger.info(f"Extracted entities: {entities}")
        
        found_in_pdf = False
        pdf_response = None
        pdf_source = None
        confidence = None
        
        # First try custom PDFs
        if custom_pdf_data:
            for pdf_id, pdf_info in custom_pdf_data.items():
                if not pdf_info.get("processed", False):
                    continue
                    
                if sentence_model and pdf_info.get("embeddings") is not None:
                    try:
                        question_embedding = sentence_model.encode(user_question, convert_to_tensor=True)
                        cos_scores = util.pytorch_cos_sim(question_embedding, pdf_info["embeddings"])[0]
                        best_idx = torch.argmax(cos_scores).item()
                        best_score = cos_scores[best_idx].item()
                        
                        # Lower threshold for better recall
                        if best_score > 0.4:
                            best_paragraph = pdf_info["paragraphs"][best_idx]
                            if qa_pipeline:
                                qa_result = qa_pipeline({
                                    'question': user_question,
                                    'context': best_paragraph
                                })
                                if qa_result['score'] > 0.1:
                                    pdf_response = qa_result['answer']
                                    confidence = qa_result['score']
                                else:
                                    pdf_response = best_paragraph
                                    confidence = best_score
                            else:
                                pdf_response = best_paragraph
                                confidence = best_score
                            
                            pdf_source = f"Source: {pdf_info['filename']} ({pdf_info['type']})"
                            found_in_pdf = True
                            break
                    except Exception as e:
                        logger.error(f"Error in semantic PDF search: {str(e)}")
                        continue
            
            if not found_in_pdf:
                for pdf_id, pdf_info in custom_pdf_data.items():
                    relevant_text = search_pdf_content(user_question, pdf_info["text"])
                    if relevant_text:
                        found_in_pdf = True
                        pdf_response = relevant_text
                        pdf_source = f"Source: {pdf_info['filename']} ({pdf_info['type']})"
                        confidence = 0.7
                        break
        
        # Return PDF response if found
        if found_in_pdf and pdf_response:
            logger.info(f"Found answer in uploaded PDF: {pdf_source}")
            return jsonify({
                "answer": pdf_response,
                "legal_reference": pdf_source,
                "confidence": confidence
            })
            
        # Try legal database with advanced matching
        legal_answer, legal_references, match_confidence = find_legal_reference(user_question)
        if legal_answer:
            return jsonify({
                "answer": legal_answer,
                "legal_reference": "References: " + ", ".join(legal_references),
                "confidence": match_confidence
            })
        
        # Fallback for specific legal topics based on keywords
        for entity in entities:
            for topic, data in legal_database.items():
                if entity in data['keywords']:
                    return jsonify({
                        "answer": data['text'],
                        "legal_reference": "References: " + ", ".join(data['references']),
                        "confidence": 0.6
                    })
            
        # For the specific case of rash driving in school zones
        if any(word in user_question.lower() for word in ["rash", "speed", "driving"]) and any(word in user_question.lower() for word in ["kill", "death", "student", "child", "school"]):
            return jsonify({
                "answer": "In cases of rash driving causing death, especially in school zones, the driver can be charged under Section 304A of the Indian Penal Code for causing death by negligence, which carries imprisonment up to 2 years, or fine, or both. Additionally, under the Motor Vehicles Act, driving at excessive speeds in school zones attracts enhanced penalties. The punishment can be more severe depending on the circumstances, such as blood alcohol content or previous traffic violations.",
                "legal_reference": "References: Indian Penal Code Section 304A, Motor Vehicles Act, Local Traffic Regulations",
                "confidence": 0.8
            })
        
        # Smart fallback responses - contextual but generic
        if any(word in user_question.lower() for word in ["driving", "accident", "vehicle", "car", "traffic"]):
            return jsonify({
                "answer": "Traffic violations in India are governed by the Motor Vehicles Act. Serious offenses like causing death by negligent driving can lead to criminal charges under Section 304A of the Indian Penal Code, with imprisonment up to 2 years. For specific legal advice on your traffic-related scenario, please consult a legal professional.",
                "legal_reference": "References: Motor Vehicles Act, Indian Penal Code",
                "confidence": 0.5
            }) 
        
        if any(word in user_question.lower() for word in ["criminal", "crime", "offense", "punishment", "jail"]):
            return jsonify({
                "answer": "Criminal offenses in India are primarily governed by the Indian Penal Code. The nature and severity of punishment depends on the specific offense, circumstances, and mitigating factors. For proper legal advice regarding criminal matters, it's advisable to consult with a criminal law attorney.",
                "legal_reference": "References: Indian Penal Code, Criminal Procedure Code",
                "confidence": 0.5
            })
        
        # Generic response as last resort
        generic_responses = [
            "I couldn't find specific information on that topic. Legal matters can be complex - please consult with a qualified legal professional for accurate advice.",
            "That's an important legal question. While I don't have specific information on this topic, I recommend consulting with a lawyer for proper guidance.",
            "I don't have enough information to answer that legal question accurately. For reliable advice, please contact a legal expert in this area."
        ]
        
        return jsonify({
            "answer": np.random.choice(generic_responses),
            "legal_reference": "No specific reference available",
            "confidence": 0.3
        })
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your request. Please try again later."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    models_loaded = {
        "qa_model": qa_pipeline is not None,
        "sentence_model": sentence_model is not None,
        "summarizer": summarizer is not None
    }
    return jsonify({
        "status": "ok",
        "message": "Service is running",
        "models_loaded": models_loaded,
        "pdf_count": len(custom_pdf_data)
    })

if __name__ == '__main__':
    logger.info("Starting Legal Q&A API server with ML integration")
    logger.info(f"PDF support enabled. Upload folder: {PDF_FOLDER}")
    logger.info(f"Loaded models: QA={qa_pipeline is not None}, Sentence={sentence_model is not None}")
    logger.info("Available legal topics: " + ", ".join(legal_database.keys()))
    app.run(host='0.0.0.0', port=5000, debug=True)