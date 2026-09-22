from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import APIConnectionError, AuthenticationError, OpenAIError

from src.rag_pipeline import answer_question
from src.vector_store import DEFAULT_INDEX_DIR


load_dotenv()

st.set_page_config(page_title="Assistant RAG Culinaire", page_icon=":material/restaurant:", layout="wide")

st.title("Assistant RAG Culinaire")
st.caption("Recherche augmentee par generation sur les volumes de recettes dans le dossier Data.")

index_dir = Path(os.getenv("FAISS_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
api_key_exists = bool(os.getenv("OPENAI_API_KEY"))
index_exists = (index_dir / "index.faiss").exists() and (index_dir / "chunks.json").exists()
proxy_vars = [key for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"] if os.getenv(key)]

with st.sidebar:
    st.header("Etat du projet")
    st.write("Cle OpenAI :", "configuree" if api_key_exists else "manquante")
    st.write("Index FAISS :", "pret" if index_exists else "a construire")
    st.write("Proxy :", "detecte" if proxy_vars else "non detecte")
    st.write("Dossier index :", str(index_dir))
    st.divider()
    top_k = st.slider("Nombre de passages recuperes", min_value=3, max_value=10, value=5)
    chat_model = st.text_input("Modele de generation", os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"))
    embedding_model = st.text_input(
        "Modele d'embeddings",
        os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )

example_questions = [
    "Donne-moi une recette avec du poulet.",
    "Trouve un dessert avec des fraises.",
    "Quelle recette contient du brocoli ?",
    "Propose une recette de boisson ou punch.",
    "Quelle recette puis-je faire avec du fromage ?",
]

selected_example = st.selectbox("Exemples de questions", [""] + example_questions)
default_question = selected_example or ""
question = st.text_area("Question", value=default_question, height=100)

if not api_key_exists:
    st.warning("Ajoute ta cle dans un fichier `.env` avec `OPENAI_API_KEY=...`.")

if not index_exists:
    st.info("L'index n'existe pas encore. Lance `python build_index.py` avant d'utiliser le chatbot.")

if proxy_vars:
    st.caption(
        "Des variables proxy sont detectees. Si l'appel OpenAI echoue, verifie ton VPN/proxy "
        "ou lance PowerShell sans ces variables."
    )

if st.button("Repondre", type="primary", disabled=not api_key_exists or not index_exists):
    if not question.strip():
        st.error("Pose une question avant de lancer la recherche.")
    else:
        with st.spinner("Recherche des passages pertinents et generation de la reponse..."):
            try:
                result = answer_question(
                    question=question,
                    index_dir=str(index_dir),
                    top_k=top_k,
                    chat_model=chat_model,
                    embedding_model=embedding_model,
                )
            except AuthenticationError:
                st.error("Authentification OpenAI refusee. Verifie la valeur de OPENAI_API_KEY dans `.env`.")
                st.stop()
            except APIConnectionError:
                st.error(
                    "Connexion impossible a l'API OpenAI. Verifie ta connexion internet, ton VPN/proxy "
                    "et les variables HTTP_PROXY/HTTPS_PROXY/ALL_PROXY."
                )
                st.code("python check_openai_connection.py", language="powershell")
                st.stop()
            except OpenAIError as exc:
                st.error(f"Erreur OpenAI : {exc}")
                st.stop()

        st.subheader("Reponse")
        st.write(result.answer)

        st.subheader("Sources utilisees")
        for i, source in enumerate(result.sources, start=1):
            chunk = source.chunk
            with st.expander(
                f"Source {i} - {chunk.volume}, page {chunk.page} - score {source.score:.3f}"
            ):
                st.write(f"Fichier : `{chunk.source_file}`")
                st.write(chunk.text)
