import os
import json
import pickle
import numpy as np
import pandas as pd
import traceback
import threading
import time
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
from dotenv import load_dotenv
# Charger les variables d'environnement dès le début
load_dotenv()
# Imports pour RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Configuration Flask
app = Flask(__name__)
CORS(app)

# =============================================
# CONFIGURATION EMAIL
# =============================================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD")
app.config['MAIL_MAX_EMAILS'] = None
app.config['MAIL_ASCII_ATTACHMENTS'] = False
app.config['MAIL_SUPPRESS_SEND'] = False
mail = Mail(app)

# =============================================
# CONFIGURATION RAG
# =============================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(CURRENT_DIR, "CHARTE_DU_CONTRIBUABLE_2025.pdf")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FAISS_INDEX_PATH = "faiss_index"

# =============================================
# VARIABLES GLOBALES
# =============================================
EMAIL_LOG_FILE = 'email_log.json'
qa_chain = None
initialization_status = {"status": "not_started", "progress": 0, "message": ""}

# =============================================
# FONCTIONS UTILITAIRES EMAIL
# =============================================
def load_email_log():
    """Charge l'historique des emails envoyés"""
    if os.path.exists(EMAIL_LOG_FILE):
        with open(EMAIL_LOG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_email_log(log):
    """Sauvegarde l'historique des emails"""
    with open(EMAIL_LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)

def log_email_sent(email, subject):
    """Enregistre l'envoi d'un email pour statistiques"""
    log = load_email_log()
    if email not in log:
        log[email] = []
    
    log[email].append({
        'date': datetime.now().isoformat(),
        'subject': subject
    })
    
    save_email_log(log)

# =============================================
# FONCTIONS PRÉDICTION FISCALE
# =============================================
def load_model_and_scaler():
    model_path = os.path.join(os.path.dirname(__file__), "model_sklearn.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    encoded_path = os.path.join(os.path.dirname(__file__), "encoded_columns.json")
    
    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        with open(encoded_path, "r") as encoded_file:
            encoded_cols = json.load(encoded_file)
        return model, scaler, encoded_cols
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None, None, None

# Chargement initial du modèle
model, scaler, encoded_cols = load_model_and_scaler()

# En-tête des données
header = ['revenu_annuel', 'retards_paiement', 'sanctions_passées', 'secteur_activité', 'chiffre_affaire', 
          'localisation', 'nb_controles', 'niveau_risque', 'prob_non_conformité']

def predict_with_artifacts(input_data_dict, threshold=0.5):
    if model is None or scaler is None or encoded_cols is None:
        raise ValueError("Modèle non chargé correctement")
    
    input_df = pd.DataFrame([input_data_dict])
    input_df['ratio_revenu_ca'] = input_df['revenu_annuel'] / input_df['chiffre_affaire'].replace(0, np.nan)
    input_df['ratio_revenu_ca'] = input_df['ratio_revenu_ca'].fillna(0)
    input_df['interaction_retards_sanctions'] = input_df['retards_paiement'] * input_df['sanctions_passées']
    
    input_df_encoded = pd.get_dummies(input_df, columns=['secteur_activité', 'localisation', 'niveau_risque'], drop_first=True)
    for col in encoded_cols:
        if col not in input_df_encoded.columns and col != "prob_non_conformité":
            input_df_encoded[col] = 0
    feature_cols = [col for col in encoded_cols if col != "prob_non_conformité"]
    input_df_encoded = input_df_encoded[feature_cols]

    input_scaled = scaler.transform(input_df_encoded)
    pred = (model.predict_proba(input_scaled)[:, 1] > threshold).astype(int)[0]
    proba = model.predict_proba(input_scaled)[0][1]
    return pred, proba

# =============================================
# FONCTIONS RAG
# =============================================
def load_pdf():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Le fichier {PDF_PATH} est introuvable.")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = PDF_PATH
    return docs

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store

def load_vector_store():
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def setup_llm():
    if not GROQ_API_KEY:
        raise ValueError("Clé API GROQ_API_KEY non définie.")
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",
        temperature=0.1
    )

def setup_rag_chain(vector_store):
    llm = setup_llm()
    prompt_template = """Vous êtes un conseiller fiscal expert au Cameroun 🇨🇲.
Répondez toujours de manière directe, claire et professionnelle aux questions fiscales.
Votre ton doit être naturel, chaleureux et confiant, reflétant une parfaite maîtrise du domaine.
Intégrez des émojis pertinents pour rendre la réponse plus conviviale, sans excès.

Si la question porte sur la fiscalité au Cameroun → répondez avec précision et concision.

Si la question est une simple politesse (ex. « Bonjour », « Comment allez-vous ? ») → répondez poliment et avec bienveillance.

Si la question sort du cadre fiscal → refusez poliment de répondre en restant courtois et positif.

Ne mentionnez jamais l existence d une source ou d un document.

Question : {question}

Informations pertinentes : {context}

Réponse : """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

def initialize_rag_system():
    global qa_chain, initialization_status
    
    try:
        initialization_status = {"status": "initializing", "progress": 10, "message": "Vérification de l'index existant..."}
        
        vector_store = load_vector_store()
        
        if vector_store is None:
            initialization_status = {"status": "initializing", "progress": 20, "message": "Chargement du PDF..."}
            documents = load_pdf()
            
            initialization_status = {"status": "initializing", "progress": 40, "message": f"{len(documents)} pages chargées. Segmentation..."}
            chunks = split_documents(documents)
            
            initialization_status = {"status": "initializing", "progress": 60, "message": f"{len(chunks)} morceaux créés. Création de l'index vectoriel..."}
            vector_store = create_vector_store(chunks)
            
            initialization_status = {"status": "initializing", "progress": 80, "message": "Index vectoriel créé."}
        else:
            initialization_status = {"status": "initializing", "progress": 60, "message": "Index existant trouvé et chargé."}
        
        initialization_status = {"status": "initializing", "progress": 90, "message": "Configuration du modèle..."}
        qa_chain = setup_rag_chain(vector_store)
        
        initialization_status = {"status": "ready", "progress": 100, "message": "Système prêt !"}
        print("✅ Système RAG initialisé avec succès!")
        
    except Exception as e:
        error_msg = f"Erreur lors de l'initialisation RAG: {str(e)}"
        initialization_status = {"status": "error", "progress": 0, "message": error_msg}
        print(f"❌ {error_msg}")
        traceback.print_exc()

# =============================================
# ROUTES API - PRÉDICTION ET EMAIL
# =============================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Aucune donnée fournie"}), 400
        
        prediction, probability = predict_with_artifacts(data)
        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/send_email', methods=['POST'])
def send_email():
    data = request.get_json()
    email = data.get('email')
    subject = data.get('subject', 'Alerte fiscale')
    body = data.get('body', '')

    if not email or not body:
        return jsonify({'error': 'Email et contenu requis'}), 400

    try:
        msg = Message(subject, recipients=[email], body=body, sender=app.config['MAIL_USERNAME'])
        mail.send(msg)
        
        log_email_sent(email, subject)
        
        return jsonify({'success': True, 'message': f'Email envoyé avec succès à {email}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/send_bulk_email', methods=['POST'])
def send_bulk_email():
    data = request.get_json()
    emails = data.get('emails', [])
    subject = data.get('subject', 'Alerte fiscale')
    body = data.get('body', '')

    if not emails or not body:
        return jsonify({'error': 'Liste d\'emails et contenu requis'}), 400

    if len(emails) > 100:
        return jsonify({'error': 'Maximum 100 emails par envoi en masse'}), 400

    results = []
    success_count = 0
    error_count = 0

    for email in emails:
        try:
            msg = Message(subject, recipients=[email], body=body, sender=app.config['MAIL_USERNAME'])
            mail.send(msg)
            log_email_sent(email, subject)
            results.append({'email': email, 'status': 'success'})
            success_count += 1
        except Exception as e:
            results.append({'email': email, 'status': 'error', 'error': str(e)})
            error_count += 1

    return jsonify({
        'success': True,
        'total_sent': len(emails),
        'success_count': success_count,
        'error_count': error_count,
        'results': results,
        'message': f'Envoi terminé : {success_count} succès, {error_count} échecs'
    })

@app.route('/email_stats', methods=['GET'])
def email_stats():
    log = load_email_log()
    now = datetime.now()
    
    total_emails = sum(len(emails) for emails in log.values())
    unique_recipients = len(log)
    
    today_emails = 0
    today = now.strftime('%Y-%m-%d')
    for email_list in log.values():
        for email_data in email_list:
            if email_data['date'].startswith(today):
                today_emails += 1
    
    return jsonify({
        'total_emails_sent': total_emails,
        'emails_today': today_emails,
        'unique_recipients': unique_recipients,
        'gmail_daily_limit': 500,
        'gmail_remaining_today': max(0, 500 - today_emails),
        'status': 'UNLIMITED_SENDING_ENABLED'
    })

# =============================================
# ROUTES API - CHATBOT RAG
# =============================================
@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(initialization_status)

@app.route('/api/initialize', methods=['POST'])
def initialize():
    global initialization_status
    if initialization_status["status"] != "initializing":
        thread = threading.Thread(target=initialize_rag_system)
        thread.daemon = True
        thread.start()
        return jsonify({"message": "Initialisation démarrée"}), 202
    else:
        return jsonify({"message": "Initialisation déjà en cours"}), 409

@app.route('/api/chat', methods=['POST'])
def chat():
    global qa_chain
    
    try:
        if initialization_status["status"] != "ready":
            return jsonify({
                "error": "Système RAG non prêt",
                "status": initialization_status["status"],
                "message": initialization_status["message"]
            }), 503
        
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Question manquante"}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({"error": "Question vide"}), 400
        
        result = qa_chain({"query": question})
        
        response = {
            "answer": result["result"],
            "sources": []
        }
        
        if data.get('include_sources', False):
            for doc in result["source_documents"]:
                response["sources"].append({
                    "source": doc.metadata.get('source', 'N/A'),
                    "page": doc.metadata.get('page', 'N/A'),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Erreur lors du traitement: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Erreur interne: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "rag_system_status": initialization_status["status"],
        "prediction_model_loaded": model is not None
    })

@app.route('/api/info', methods=['GET'])
def get_info():
    return jsonify({
        "pdf_path": PDF_PATH,
        "pdf_exists": os.path.exists(PDF_PATH),
        "faiss_index_exists": os.path.exists(FAISS_INDEX_PATH),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "prediction_model_loaded": model is not None,
        "features": [
            "Prédiction fiscale",
            "Envoi d'emails (simple et en masse)",
            "Chatbot RAG pour questions fiscales",
            "Statistiques d'emails"
        ]
    })

# =============================================
# GESTION DES ERREURS
# =============================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint non trouvé"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Erreur interne du serveur"}), 500

# =============================================
# INITIALISATION ET DÉMARRAGE
# =============================================
if __name__ == "__main__":
    print("🚀 Démarrage de l'API Flask intégrée...")
    print("📊 Modèle de prédiction:", "✅ Chargé" if model else "❌ Erreur")
    
    # Initialiser le système RAG dans un thread séparé
    if os.path.exists(PDF_PATH):
        init_thread = threading.Thread(target=initialize_rag_system)
        init_thread.daemon = True
        init_thread.start()
        print("🤖 Initialisation du système RAG en cours...")
    else:
        print("⚠️  PDF non trouvé, système RAG désactivé")
    
    print("\n🌐 API disponible sur http://localhost:5000")
    print("📋 Endpoints disponibles:")
    print("  === PRÉDICTION ET EMAIL ===")
    print("  - POST /predict           : Prédiction fiscale")
    print("  - POST /send_email        : Envoyer un email")
    print("  - POST /send_bulk_email   : Envoi d'emails en masse")
    print("  - GET  /email_stats       : Statistiques d'emails")
    print("  === CHATBOT RAG ===")
    print("  - GET  /api/status        : Statut du système RAG")
    print("  - POST /api/initialize    : Réinitialiser le système RAG")
    print("  - POST /api/chat          : Poser une question")
    print("  === GÉNÉRAL ===")
    print("  - GET  /api/health        : Vérification de santé")
    print("  - GET  /api/info          : Informations système")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)