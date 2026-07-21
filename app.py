import streamlit as st
import torch
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

from knowledge_base import KNOWLEDGE_BASE

# ============================================================
# DETECTION DE LANGUE ET TRADUCTION
# ============================================================

MOTS_FRANCAIS = {
    "quelle", "quel", "quels", "quelles", "qu'est-ce", "qu est ce",
    "comment", "pourquoi", "combien", "où", "ou est", "qui est",
    "est-ce", "peux-tu", "peux tu", "explique", "explique-moi",
    "définis", "définir", "c'est quoi", "c est quoi", "quelles sont",
    "quels sont", "je veux savoir", "dis-moi", "dis moi",
}


def est_francais(question):
    """Détecte si la question est posée en français (heuristique simple)."""
    q = question.lower()
    if any(c in q for c in "éèêàùçôî"):
        return True
    if any(mot in q for mot in MOTS_FRANCAIS):
        return True
    return False


def traduire_en_francais(texte_anglais):
    """Traduit un texte anglais en français. En cas d'échec, renvoie le texte original + note."""
    try:
        return GoogleTranslator(source="en", target="fr").translate(texte_anglais)
    except Exception:
        return texte_anglais + "\n\n*(Traduction indisponible, réponse affichée en anglais)*"


# ============================================================
# TRADUCTEUR (fonctionnalité dédiée, prioritaire)
# ============================================================

MOTS_TRADUCTION = ["traduire", "traduis", "traduction de", "translate"]

# ============================================================
# DETECTION DE CONVERSION DE DEVISES (hors de portée, réponse honnête)
# ============================================================

MOTS_DEVISES = [
    "dinar", "euro", "dollar", "livre sterling", "franc suisse", "yen",
    "usd", "eur", "gbp", "tnd", "cad", "taux de change", "exchange rate",
    "combien vaut", "convertir en", "conversion de devise",
]


def est_demande_devise(question):
    """Détecte une demande de conversion de devises (impossible à répondre de façon fiable et à jour)."""
    q = question.lower()
    nb_mots_devise = sum(1 for m in MOTS_DEVISES if m in q)
    return nb_mots_devise >= 2  # au moins 2 mentions de devises (ex: dinar + euro)


def detecter_demande_traduction(question):
    """Détecte une demande de traduction et extrait le texte + la langue cible."""
    q = question.lower()
    if not any(m in q for m in MOTS_TRADUCTION):
        return None

    if any(m in q for m in ["en anglais", "into english", "to english", "in english"]):
        cible = "en"
    elif any(m in q for m in ["en français", "en francais", "into french", "to french", "in french"]):
        cible = "fr"
    else:
        cible = None  # déduite automatiquement selon la langue détectée du texte

    if ":" in question:
        texte = question.split(":", 1)[1].strip()
    else:
        texte = question
        a_retirer = MOTS_TRADUCTION + ["en anglais", "en français", "en francais", "into english",
                                        "into french", "cette phrase", "ce texte", "this sentence", "?"]
        for m in a_retirer:
            texte = re.sub(re.escape(m), "", texte, flags=re.IGNORECASE)
        texte = texte.strip(" :.-")

    if not texte:
        return None
    return texte, cible


def executer_traduction(texte, cible):
    """Traduit un texte donné vers la langue cible (déduite automatiquement si None)."""
    try:
        if cible is None:
            cible = "en" if est_francais(texte) else "fr"
        source = "fr" if cible == "en" else "en"
        traduction = GoogleTranslator(source=source, target=cible).translate(texte)
        langue_nom = {"en": "anglais", "fr": "français"}[cible]
        return f'"{texte}" → **{traduction}** ({langue_nom})'
    except Exception:
        return "Désolé, la traduction n'a pas pu être effectuée (problème de connexion)."


# ============================================================
# CALCULATEUR ARITHMÉTIQUE (priorité absolue avant l'IA)
# ============================================================
# Les maths simples ne doivent JAMAIS être laissées à GPT-2, qui n'est
# pas fiable pour ça. On calcule directement avec Python, précis même
# pour de très grands nombres.

def calculer_expression(question):
    """Détecte et calcule une opération arithmétique simple (+, -, *, /)."""
    q = question.replace(" ", "")
    match = re.search(r'(-?\d+\.?\d*)([\+\-\*x×/])(-?\d+\.?\d*)', q)
    if not match:
        return None

    a_str, op, b_str = match.groups()
    a = int(a_str) if "." not in a_str else float(a_str)
    b = int(b_str) if "." not in b_str else float(b_str)

    try:
        if op == "+":
            resultat = a + b
        elif op == "-":
            resultat = a - b
        elif op in ("*", "x", "×"):
            resultat = a * b
        elif op == "/":
            if b == 0:
                return "Division by zero is undefined."
            resultat = a / b
        else:
            return None
    except Exception:
        return None

    op_disp = "×" if op in ("*", "x", "×") else op
    return f"{a} {op_disp} {b} = {resultat}"


# ============================================================
# MINI BASE DE FAITS VÉRIFIÉS (capitales du monde)
# ============================================================
# Approche "RAG léger" : pour les questions de capitales, on répond
# depuis une source fiable plutôt que de laisser GPT-2 inventer.

CAPITALES = {
    "france": "Paris", "tunisia": "Tunis", "tunisie": "Tunis",
    "morocco": "Rabat", "maroc": "Rabat", "algeria": "Algiers",
    "algerie": "Algiers", "libya": "Tripoli", "egypt": "Cairo",
    "egypte": "Cairo", "germany": "Berlin", "allemagne": "Berlin",
    "italy": "Rome", "italie": "Rome", "spain": "Madrid",
    "espagne": "Madrid", "portugal": "Lisbon", "uk": "London",
    "united kingdom": "London", "england": "London",
    "angleterre": "London", "usa": "Washington, D.C.",
    "united states": "Washington, D.C.", "etats-unis": "Washington, D.C.",
    "canada": "Ottawa", "china": "Beijing", "chine": "Beijing",
    "japan": "Tokyo", "japon": "Tokyo", "india": "New Delhi",
    "inde": "New Delhi", "brazil": "Brasilia", "bresil": "Brasilia",
    "russia": "Moscow", "russie": "Moscow", "turkey": "Ankara",
    "turquie": "Ankara", "greece": "Athens", "grece": "Athens",
    "switzerland": "Bern", "suisse": "Bern", "belgium": "Brussels",
    "belgique": "Brussels", "netherlands": "Amsterdam",
    "pays-bas": "Amsterdam", "senegal": "Dakar", "mali": "Bamako",
    "ivory coast": "Yamoussoukro", "cote d'ivoire": "Yamoussoukro",
    "saudi arabia": "Riyadh", "arabie saoudite": "Riyadh",
    "qatar": "Doha", "uae": "Abu Dhabi",
    "emirats arabes unis": "Abu Dhabi",
}


def chercher_capitale(question):
    """Vérifie si la question porte sur une capitale connue et fiable."""
    q = question.lower()
    if "capital" not in q:
        return None
    for pays, capitale in CAPITALES.items():
        if pays in q:
            return f"The capital of {pays.title()} is {capitale}."
    return None


# ============================================================
# MOTEUR RAG (recherche dans la base de connaissances multi-matières)
# ============================================================

SEUIL_SIMILARITE = 0.15  # en dessous de ce score, on considère qu'il n'y a pas de bon match

@st.cache_resource
def construire_index_rag():
    """Prépare le moteur de recherche (TF-IDF) sur la base de connaissances."""
    textes = [f"{doc['title']} {doc['content']}" for doc in KNOWLEDGE_BASE]
    stop_words_etendus = list(ENGLISH_STOP_WORDS) + [
        "define", "defined", "defining", "definition",
        "explain", "explains", "explained", "explaining",
        "describe", "describes", "described", "describing",
        "tell", "tells", "telling",
    ]
    vectorizer = TfidfVectorizer(stop_words=stop_words_etendus)
    matrix = vectorizer.fit_transform(textes)
    return vectorizer, matrix


def chercher_dans_base(question, vectorizer, matrix):
    """Trouve la fiche la plus pertinente pour la question, si elle existe."""
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix)[0]
    idx = scores.argmax()
    if scores[idx] >= SEUIL_SIMILARITE:
        return KNOWLEDGE_BASE[idx], scores[idx]
    return None, scores[idx]


def generer_reponse_avec_contexte(model, tokenizer, question, contexte, device, max_length=100, temperature=0.4):
    """Génère une réponse en s'appuyant sur un passage de la base de connaissances (RAG)."""
    model.eval()
    prompt = f"Context: {contexte}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.85,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    reponse = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reponse = reponse.split("Answer:")[-1].strip()
    phrases = re.split(r'(?<=[.!?])\s+', reponse)
    reponse_courte = " ".join(phrases[:2]).strip()
    return reponse_courte if reponse_courte else reponse


# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(page_title="Chatbot ENET'Com - GPT-2 Fine-tuné", page_icon="🤖")

# ============================================================
# CHARGEMENT DU MODÈLE (mis en cache pour la rapidité)
# ============================================================

MODEL_NAME = "Aicha83/chatbot-gpt2-finetuned"  # ton modèle hébergé sur Hugging Face

@st.cache_resource
def charger_modele():
    device = torch.device("cpu")  # Streamlit Cloud n'a pas de GPU

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    return model, tokenizer, device


# ============================================================
# FONCTION DE GÉNÉRATION DE RÉPONSE (sans contexte, cas par défaut)
# ============================================================

def generer_reponse(model, tokenizer, question, device, max_length=80, temperature=0.4):
    model.eval()
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.85,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    reponse = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reponse = reponse.split("Answer:")[-1].strip()
    phrases = re.split(r'(?<=[.!?])\s+', reponse)
    reponse_courte = " ".join(phrases[:2]).strip()
    return reponse_courte if reponse_courte else reponse


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.title("🤖 Chatbot GPT-2 Fine-tuné + RAG")
st.caption("Projet de stage — ESSE Lab, ENET'Com Sfax — GPT-2 base fine-tuné + recherche documentaire (RAG)")

with st.expander("ℹ️ À propos de ce chatbot"):
    st.markdown(f"""
    Ce chatbot utilise **GPT-2 base**, un modèle pré-entraîné par OpenAI (124M paramètres),
    **fine-tuné** sur le dataset Databricks Dolly 15k, combiné à un système **RAG**
    (Retrieval-Augmented Generation).

    **Comment ça marche :**
    1. La question est comparée automatiquement à une base de **{len(KNOWLEDGE_BASE)} fiches**
       (maths, physique, géographie, civilisation, électronique, informatique, cybersécurité, chimie)
    2. Si une fiche pertinente est trouvée → la réponse **vérifiée** de cette fiche est affichée directement (RAG extractif)
    3. Les questions de capitales sont répondues directement depuis une base fiable
    4. Sinon → GPT-2 génère librement (moins fiable, signalé par un avertissement)

    **Pourquoi un RAG "extractif" plutôt que "génératif" ?**
    Des tests ont montré que GPT-2 base, n'étant pas entraîné à l'instruction-following,
    n'arrive pas à reformuler fidèlement un contexte fourni — il continue à halluciner
    même avec la bonne information sous les yeux. Afficher directement le contenu vérifié
    de la fiche garantit donc une réponse fiable pour les sujets couverts par la base.

    **Caractéristiques techniques :**
    - Modèle de base : GPT-2 (124M paramètres)
    - Fine-tuning : 8 époques sur ~8500 exemples (Dolly 15k)
    - Recherche : TF-IDF + similarité cosinus sur la base de connaissances
    - Génération : anti-répétition activée (repetition_penalty, no_repeat_ngram_size)

    ⚠️ **Limites connues** : la base de connaissances est volontairement limitée (démonstration
    de projet de stage). Pour les sujets non couverts, le modèle peut encore halluciner —
    ce comportement illustre concrètement l'intérêt d'un RAG à plus grande échelle pour
    un chatbot de production fiable.
    """)

# Charger le modèle et l'index RAG (une seule fois, mis en cache)
try:
    model, tokenizer, device = charger_modele()
    vectorizer, matrix = construire_index_rag()
    modele_charge = True
except Exception as e:
    modele_charge = False
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.info(f"Vérifie que le modèle `{MODEL_NAME}` est bien public sur Hugging Face et accessible.")

if modele_charge:
    # Initialiser l'historique de conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Afficher l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Zone de saisie utilisateur
    if question := st.chat_input("Pose ta question ici (en anglais)..."):
        # Afficher la question de l'utilisateur
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Générer et afficher la réponse
        with st.chat_message("assistant"):
            with st.spinner("Génération de la réponse..."):
                francais = est_francais(question)

                # Si la question est en français, on la traduit en anglais pour la recherche
                # (la base de connaissances est en anglais) — on garde la question originale pour l'affichage
                question_recherche = question
                if francais:
                    try:
                        question_recherche = GoogleTranslator(source="fr", target="en").translate(question)
                    except Exception:
                        question_recherche = question  # si la traduction échoue, on cherche avec le texte original

                # 0. Vérifier d'abord si c'est un calcul arithmétique (priorité absolue)
                reponse_calcul = calculer_expression(question)
                demande_trad = detecter_demande_traduction(question)
                demande_devise = est_demande_devise(question)

                if demande_devise:
                    reponse = ("Je ne peux pas faire de conversion de devises fiable, car les taux de change "
                               "changent en temps réel et je n'ai pas accès à des données à jour. "
                               "Utilise un convertisseur en ligne (comme Google, XE.com, ou ton application bancaire) "
                               "pour un taux exact et actuel.")
                    if francais:
                        pass  # déjà en français
                    else:
                        reponse = ("I can't provide reliable currency conversion, since exchange rates change "
                                   "in real time and I don't have access to live data. "
                                   "Please use an online converter (like Google, XE.com, or your banking app) "
                                   "for an exact, current rate.")
                    badge = "⚠️ Hors de portée (données en temps réel non disponibles)"
                elif reponse_calcul:
                    reponse = reponse_calcul
                    badge = "🧮 Calcul exact (Python)"
                # 0.5 Vérifier si c'est une demande de traduction
                elif demande_trad:
                    texte, cible = demande_trad
                    reponse = executer_traduction(texte, cible)
                    badge = "🌐 Traduction"
                # 1. Vérifier ensuite si c'est une question de capitale
                elif chercher_capitale(question_recherche):
                    reponse = chercher_capitale(question_recherche)
                    if francais:
                        reponse = traduire_en_francais(reponse)
                    badge = "✅ Réponse vérifiée (base de capitales)"
                else:
                    # 2. Chercher dans la base de connaissances multi-matières
                    # On essaie à la fois la version traduite ET la version originale
                    # (la traduction peut corrompre des acronymes techniques comme "MOSFET", "TCP", etc.)
                    doc_trad, score_trad = chercher_dans_base(question_recherche, vectorizer, matrix)
                    if francais:
                        doc_brut, score_brut = chercher_dans_base(question, vectorizer, matrix)
                        # On ignore un match "brut" venant des fiches de grammaire (French/English) :
                        # elles captent à tort des mots français/anglais génériques sans rapport avec le sujet
                        if doc_brut and doc_brut["subject"] in ("French", "English"):
                            doc_brut, score_brut = None, 0
                        if doc_brut and score_brut > score_trad:
                            doc_trouve, score = doc_brut, score_brut
                        else:
                            doc_trouve, score = doc_trad, score_trad
                    else:
                        doc_trouve, score = doc_trad, score_trad

                    if doc_trouve:
                        reponse = doc_trouve["content"]
                        if francais:
                            reponse = traduire_en_francais(reponse)
                        badge = f"📚 Réponse vérifiée : *{doc_trouve['title']}* ({doc_trouve['subject']}) — RAG"
                    else:
                        # 3. Aucun document pertinent -> génération libre (toujours en anglais chez GPT-2)
                        reponse = generer_reponse(model, tokenizer, question_recherche, device)
                        if francais:
                            reponse = traduire_en_francais(reponse)
                        badge = "🤖 Réponse générée par GPT-2 (non vérifiée — aucun document pertinent trouvé)"

                st.caption(badge)
            st.markdown(reponse)
        st.session_state.messages.append({"role": "assistant", "content": reponse})

    # Bouton pour réinitialiser la conversation
    if st.session_state.messages:
        if st.button("🗑️ Réinitialiser la conversation"):
            st.session_state.messages = []
            st.rerun()
