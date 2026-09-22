# Assistant RAG culinaire

Assistant de recettes basé sur une architecture **RAG** (*Retrieval-Augmented Generation*). Le projet extrait le texte de documents PDF, le découpe en passages, crée des embeddings avec OpenAI, puis utilise FAISS pour retrouver les passages pertinents avant de générer une réponse citant ses sources.

L'application fournit une interface Streamlit permettant de poser des questions en français sur le corpus de recettes.

## Fonctionnement

```text
PDF dans Data/
	-> extraction et nettoyage du texte
	-> découpage en chunks
	-> embeddings OpenAI
	-> index vectoriel FAISS
	-> recherche des passages pertinents
	-> réponse générée par OpenAI avec citations
```

Le projet met en pratique :

- l'ingestion de documents PDF ;
- le nettoyage et le découpage en chunks ;
- la génération d'embeddings ;
- la recherche vectorielle avec FAISS ;
- l'orchestration d'un pipeline RAG ;
- l'évaluation qualitative des réponses ;
- la création d'une interface de démonstration avec Streamlit.

## Prérequis

- Python 3.10 ou version supérieure ;
- une clé API OpenAI ;
- un corpus de fichiers PDF placé dans `Data/`.

Les fichiers PDF et l'index FAISS sont ignorés par Git. Tu dois donc fournir ton propre corpus et reconstruire l'index après avoir cloné le projet. Vérifie que tu as le droit d'utiliser et de redistribuer les documents ajoutés à `Data/`.

## Installation

Dans PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Ouvre ensuite `.env` et renseigne ta clé API :

```text
OPENAI_API_KEY=ta_cle_api
```

Les modèles utilisés par défaut sont définis dans `.env` :

```text
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
FAISS_INDEX_DIR=storage/faiss_index
OPENAI_IGNORE_PROXY=1
```

Ne publie jamais `.env` et ne place jamais une clé API directement dans le code ou dans les sorties du notebook.

## Construire l'index

Place les PDF dans `Data/`, puis exécute :

```powershell
python build_index.py
```

Le script extrait les pages, crée les chunks, appelle l'API d'embeddings et sauvegarde les fichiers `index.faiss` et `chunks.json` dans `storage/faiss_index/`.

Tu peux modifier les paramètres principaux :

```powershell
python build_index.py --data-dir Data --index-dir storage/faiss_index --chunk-size 800 --overlap 150
```

La construction de l'index et les questions posées à l'application peuvent entraîner des coûts d'utilisation de l'API OpenAI.

## Lancer l'application

Après avoir construit l'index :

```powershell
python -m streamlit run app.py
```

Dans l'interface, sélectionne une question d'exemple ou écris ta propre question. Quelques exemples :

- Donne-moi une recette avec du poulet.
- Trouve un dessert avec des fraises.
- Quelle recette contient du brocoli ?
- Propose une recette de boisson ou de punch.
- Quelle recette puis-je faire avec du fromage ?

Si l'appel OpenAI échoue, vérifie la valeur de `OPENAI_API_KEY`, ta connexion réseau et les variables proxy de ton environnement. Avec `OPENAI_IGNORE_PROXY=1`, le client ignore les variables proxy système.

## Notebook

Le notebook [01_rag_pas_a_pas.ipynb](notebooks/01_rag_pas_a_pas.ipynb) présente le pipeline étape par étape :

1. préparation du corpus ;
2. extraction du texte des PDF ;
3. découpage en chunks ;
4. création des embeddings ;
5. indexation avec FAISS ;
6. recherche de passages ;
7. génération d'une réponse avec citations.

Avant de publier le notebook, vide ses sorties afin de ne pas exposer de chemins locaux ou de contenu extrait des documents.

## Structure du projet

```text
.
├── app.py                 # Interface Streamlit
├── build_index.py         # Construction de l'index FAISS
├── requirements.txt       # Dépendances Python
├── .env.example           # Exemple de configuration
├── src/
│   ├── ingestion.py       # Extraction du texte des PDF
│   ├── chunking.py        # Découpage en passages
│   ├── vector_store.py    # Embeddings et recherche FAISS
│   ├── openai_client.py   # Client OpenAI
│   └── rag_pipeline.py    # Pipeline de génération
└── notebooks/             # Démonstration pédagogique
```

Les dossiers `Data/` et `storage/` contiennent des ressources générées ou locales et sont exclus du dépôt par `.gitignore`.

## Limites connues

- La qualité des réponses dépend des passages retrouvés par FAISS.
- Le modèle peut produire une réponse inexacte ; les sources doivent être vérifiées.
- La qualité de l'extraction dépend de la mise en page des PDF.
- Les documents sources doivent être suffisamment complets pour répondre aux questions.
