import streamlit as st
import torch
import torch.nn as nn
import math
from tokenizers import Tokenizer

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(page_title="Chatbot ENET'Com - Transformer from scratch", page_icon="🤖")

# ============================================================
# ARCHITECTURE DU TRANSFORMER (identique à celle codée dans Colab)
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=350):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        return output, attention_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=8, num_layers=4, d_ff=512, max_len=350, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_mask=None):
        x = self.embedding(src)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        output = self.output_layer(decoder_output)
        return output


# ============================================================
# CHARGEMENT DU MODÈLE ET DU TOKENIZER (mis en cache pour la rapidité)
# ============================================================

@st.cache_resource
def charger_modele():
    device = torch.device("cpu")  # Streamlit Cloud n'a pas de GPU, on utilise le CPU

    tokenizer = Tokenizer.from_file("tokenizer_chatbot_large.json")
    vocab_size = tokenizer.get_vocab_size()

    model = Transformer(vocab_size, d_model=128, num_heads=8, num_layers=4, d_ff=512, max_len=350, dropout=0.1)
    model.load_state_dict(torch.load("best_model_large.pt", map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    return model, tokenizer, device


# ============================================================
# FONCTION DE GÉNÉRATION DE RÉPONSE
# ============================================================

def generer_reponse(model, tokenizer, question, device, max_len=50):
    model.eval()

    src = torch.tensor([tokenizer.encode(question).ids]).to(device)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)

    encoder_output = model.encode(src, src_mask)

    start_id = tokenizer.token_to_id("<start>")
    end_id = tokenizer.token_to_id("<end>")
    tgt = torch.tensor([[start_id]]).to(device)

    for _ in range(max_len):
        tgt_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1))).bool().unsqueeze(0).unsqueeze(0).to(device)
        decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        output = model.output_layer(decoder_output)

        next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)
        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == end_id:
            break

    tokens_generes = tgt[0].tolist()[1:]
    if len(tokens_generes) > 0 and tokens_generes[-1] == end_id:
        tokens_generes = tokens_generes[:-1]

    if len(tokens_generes) == 0:
        return "(le modèle n'a pas généré de réponse pour cette question)"

    return tokenizer.decode(tokens_generes)


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.title("🤖 Chatbot Transformer From Scratch")
st.caption("Projet de stage — ESSE Lab, ENET'Com Sfax — Transformer codé et entraîné from scratch en PyTorch")

with st.expander("ℹ️ À propos de ce chatbot"):
    st.markdown("""
    Ce chatbot utilise une architecture **Transformer Encoder-Decoder codée entièrement from scratch** en PyTorch,
    sans utiliser de modèle pré-entraîné.

    **Caractéristiques techniques :**
    - Architecture : Transformer (Encoder-Decoder), 4 couches, 8 têtes d'attention
    - Paramètres : ~3.9 millions
    - Tokenizer : BPE (Byte-Pair Encoding), vocabulaire de 8000 sous-mots
    - Dataset d'entraînement : Databricks Dolly 15k (8578 exemples après filtrage)
    - Perplexité sur test set : ~304

    ⚠️ **Limites connues** : étant donné les ressources limitées (peu de données, peu de paramètres comparé
    aux LLM comme GPT/Claude), les réponses peuvent être répétitives ou imprécises. Ce projet a un objectif
    pédagogique et démonstratif dans le cadre d'un stage.
    """)

# Charger le modèle (une seule fois, mis en cache)
try:
    model, tokenizer, device = charger_modele()
    modele_charge = True
except Exception as e:
    modele_charge = False
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.info("Vérifie que les fichiers `best_model_large.pt` et `tokenizer_chatbot_large.json` sont bien présents dans le dossier du projet.")

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
                reponse = generer_reponse(model, tokenizer, question.lower(), device)
            st.markdown(reponse)
        st.session_state.messages.append({"role": "assistant", "content": reponse})

    # Bouton pour réinitialiser la conversation
    if st.session_state.messages:
        if st.button("🗑️ Réinitialiser la conversation"):
            st.session_state.messages = []
            st.rerun()
