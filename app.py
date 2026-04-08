import streamlit as st
import os
import json
import re
import time
import plotly.graph_objects as go
from groq import Groq
from difflib import SequenceMatcher

# Load .env file into environment if present (simple, no external deps)
def _load_dotenv(path=".env"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
    except FileNotFoundError:
        pass


_load_dotenv()
st.set_page_config(
    page_title="FallahTech RAG - Scoring Investissement",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4FC3F7, #29B6F6, #0288D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #90A4AE;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1A1F2E, #252B3B);
        border: 1px solid #2A3142;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    .score-big {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
    }
    .risk-low { color: #66BB6A; }
    .risk-medium { color: #FFA726; }
    .risk-high { color: #EF5350; }
    .source-tag {
        background: #1A237E;
        color: #90CAF9;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 2px 4px;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
    }
    .analysis-box {
        background: #1A1F2E;
        border-left: 4px solid #4FC3F7;
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        line-height: 1.7;
    }
    .criteria-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #1A237E, #283593);
        border: 2px solid #3F51B5;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .pipeline-step {
        background: #1A1F2E;
        border: 1px solid #2A3142;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

SYSTEM_PROMPT = """Tu es un analyste financier senior d'un fonds d'investissement franco-tunisien. Tu instruis le dossier Série A de FallahTech SARL — startup AgriTech tunisienne basée à Sousse.
RÈGLES ABSOLUES :
(1) JAMAIS inventer de données. Cite UNIQUEMENT les documents fournis.
(2) Si une info n'est PAS dans les documents, écris « Non disponible dans le corpus ». N'invente JAMAIS de biographies, années d'expérience, diplômes ou qualifications.
(3) Cite TOUJOURS [SOURCE: nom_fichier] après chaque fait.
(4) Chiffres EXACTS en TND, tels que fournis dans le Référentiel Vérifié.
(5) FallahTech = AgriTech, PAS télécommunications.
(6) Réponds en français professionnel.
(7) NE répète JAMAIS le même paragraphe ou la même idée deux fois. Sois concis et structuré.
(8) Respecte STRICTEMENT le format demandé. Commence directement par le format obligatoire."""

SCORING_CRITERIA = {
    "Santé Financière": {
        "weight": 0.40,
        "query": "Analyse financière FallahTech : chiffre d'affaires, résultat net, EBITDA, marge brute, trésorerie, ratio courant, solvabilité, évolution 2023-2025, rentabilité, dettes fournisseurs",
        "icon": "💰",
        "description": "Solidité financière, rentabilité, liquidité et solvabilité sur 3 exercices"
    },
    "Traction Commerciale": {
        "weight": 0.30,
        "query": "Traction commerciale FallahTech : croissance chiffre d'affaires, nombre abonnés, taux de rétention, coopératives partenaires, expansion géographique, gouvernorats couverts",
        "icon": "📈",
        "description": "Croissance du CA, base clients, rétention et dynamique commerciale"
    },
    "Qualité de l'Équipe": {
        "weight": 0.15,
        "query": "Équipe FallahTech : effectif, organigramme, compétences clés, CEO CTO, départements, salaires, recrutements, profils techniques et agronomiques",
        "icon": "👥",
        "description": "Compétences, structure, expérience et capacité d'exécution de l'équipe"
    },
    "Opportunité de Marché": {
        "weight": 0.15,
        "query": "Marché AgriTech FallahTech : TAM SAM SOM, concurrence, positionnement prix, expansion Maghreb Algérie Maroc, avantage compétitif, plan Génération Green",
        "icon": "🌍",
        "description": "Taille du marché, positionnement concurrentiel et potentiel de croissance"
    }
}

MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
]

# Comprehensive verified reference — pypdf extraction lost ALL digits from
# every PDF. This reference restores the complete data for the LLM.
CORPUS_REFERENCE = """=== RÉFÉRENTIEL VÉRIFIÉ — FALLAHTECH SARL ===
[SOURCE: Etats_Financiers_Historiques + BusinessPlan_Complet + Registre_Personnel + Etude_Marche + Statuts]

━━━ DONNÉES FINANCIÈRES ━━━

CHIFFRE D'AFFAIRES (TND):
- 2023: 250 000 TND
- 2024: 780 000 TND (croissance +212.0% vs 2023)
- 2025: 1 650 000 TND (croissance +111.5% vs 2024)
- 2026 (prévision): 2 520 000 TND
- 2027 (prévision): 3 657 000 TND
- 2028 (prévision): 5 118 000 TND
- 2029 (prévision): 6 910 000 TND

RÉSULTAT NET (TND):
- 2023: -50 000 TND (perte)
- 2024: -40 000 TND (perte)
- 2025: 45 000 TND (1ère rentabilité)
- 2026 (prévision): 321 000 TND
- 2027 (prévision): 701 000 TND

CHARGES D'EXPLOITATION (TND):
- 2023: 300 000 TND | 2024: 820 000 TND | 2025: 1 605 000 TND

EBITDA (TND):
- 2025: 180 000 TND | 2026: 420 000 TND | 2027: 810 000 TND
- 2028: 1 330 000 TND | 2029: 1 820 000 TND

MARGES:
- Marge brute: 2023: 60% | 2024: 64% | 2025: 70%
- Marge d'exploitation: 2023: -20.0% | 2024: -5.1% | 2025: 2.7%
- Marge nette: 2025: 2.7% | 2026: 12.7% | 2027: 19.2%
- Marge EBITDA: 2025: 10.9% | 2026: 16.7% | 2027: 22.1%

TRÉSORERIE (TND):
- 2023: 100 000 TND | 2024: 220 000 TND | 2025: 510 000 TND (+290K vs 2024)

RATIOS FINANCIERS:
- Ratio courant: 2024: 9.43 | 2025: 1.56
- Ratio d'endettement: 2024: 0.07 | 2025: 0.86
- ROE: 2024: -8.6% | 2025: 9.0%
- ROA: 2024: -8.0% | 2025: 4.8%
- Croissance CA: 2024: +212% | 2025: +111.5%

DETTES FOURNISSEURS: 2025: 300 000 TND (x15 vs 2024 — Alerte)

RISQUES FINANCIERS IDENTIFIÉS:
- Dépendance à la croissance du marché AgriTech
- Risque de concentration client (5 coopératives = 60% du CA)
- Risque technologique (concurrence d'acteurs internationaux)
- Dettes fournisseurs en forte hausse (300 000 TND, x15)
- Ratio courant en baisse (9.43 → 1.56)

━━━ MÉTRIQUES SaaS ━━━
- Utilisateurs actifs: 2025: 3 500 | 2026: 5 250 | 2027: 7 613 | 2028: 10 658 | 2029: 14 389
- Croissance utilisateurs: 2026: +50% | 2027: +45% | 2028: +40% | 2029: +35%
- ARPU annuel: 2025: 470 TND | 2026: 480 TND | 2027: 504 TND
- CAC (Coût d'acquisition client): 500 TND
- LTV (Lifetime Value): 2 640 TND
- LTV/CAC ratio: 5.28
- Churn mensuel: 1.5%, Taux de rétention annuel: 82%
- Payback CAC: 14 mois
- Durée de vie client: 5.5 ans

━━━ ÉQUIPE — 18 EMPLOYÉS ━━━
[SOURCE: Registre_Personnel]
ID1  | Direction   | CEO (Sami BEN YOUSSEF)        | Jan-2023 | 72 000 TND/an
ID2  | Tech        | CTO (Amira TRABELSI)          | Jan-2023 | 66 000 TND/an
ID3  | Opérations  | Head of Field Ops             | Mar-2023 | 42 000 TND/an
ID4  | Tech        | Senior Backend Dev            | Juin-2023| 48 000 TND/an
ID5  | Tech        | Mobile Dev (Android)          | Sep-2023 | 36 000 TND/an
ID6  | Tech        | Mobile Dev (iOS)              | Sep-2023 | 36 000 TND/an
ID7  | Agro        | Lead Agronome                 | Jan-2024 | 36 000 TND/an
ID8  | Sales       | Field Sales Rep (Nord)        | Fév-2024 | 24 000 TND/an + Var
ID9  | Sales       | Field Sales Rep (Centre)      | Fév-2024 | 24 000 TND/an + Var
ID10 | Tech        | Data Scientist                | Avr-2024 | 48 000 TND/an
ID11 | Agro        | Agronome Junior               | Juin-2024| 24 000 TND/an
ID12 | Tech        | UI/UX Designer                | Sep-2024 | 30 000 TND/an
ID13 | Tech        | QA Engineer                   | Nov-2024 | 30 000 TND/an
ID14 | Sales       | Field Sales Rep (Cap Bon)     | Jan-2025 | 24 000 TND/an + Var
ID15 | Sales       | Field Sales Rep (Sud)         | Mar-2025 | 24 000 TND/an + Var
ID16 | Agro        | Agronome Junior               | Mai-2025 | 24 000 TND/an
ID17 | Agro        | Agronome Junior               | Sep-2025 | 24 000 TND/an
ID18 | Sales       | Customer Success Agent        | Oct-2025 | 22 000 TND/an
Répartition: 7 Tech, 5 Sales, 4 Agro, 1 Direction, 1 Ops
Masse salariale annuelle (hors variables): 630 000 TND

━━━ MARCHÉ AGRITECH MAGHREB ━━━
[SOURCE: Etude_Marche_Synthese]
MARCHÉ TUNISIEN:
- TAM (Taille totale): 516 000 exploitations agricoles
- SAM (Marché adressable): 120 000 exploitations (connectivité smartphone + cultures adaptées)
- Part de marché actuelle: ~3% du SAM
- Concurrence locale: Faible (initiatives étatiques, solutions EU non adaptées)

POSITIONNEMENT PRIX:
- FallahTech: 15-30 TND/mois (très accessible pour petit exploitant)
- Solutions importées (xFarm, Cropin): >200 TND/mois (grands domaines)
- ROI agriculteur: estimé à 6 mois (économies d'eau et d'intrants)

EXPANSION MAGHREB:
- Algérie: 1,5 million d'exploitations, fort soutien étatique, adaptation dialecte Darja (similaire à 80%)
- Maroc: 1,5 million d'exploitations, marché le plus mature, Plan Génération Green, adaptation Darija

━━━ STRUCTURE JURIDIQUE ━━━
[SOURCE: Statuts_FallahTech]
- Forme: SARL
- Siège: Pôle Technologique de Sousse, Novation City, 4000 Sousse, Tunisie
- Capital social: 500 000 TND (5 000 parts de 100 TND)
- Actionnaires: Sami BEN YOUSSEF (CEO) 40%, Amira TRABELSI (CTO) 35%, Seed Fund AgriVentures TN 25%
- Objet: Solutions logicielles et IA appliquées à l'agriculture (AgriTech)
- Durée: 99 années
- Création: 1er Janvier 2023
VALORISATION: Pre-money 12M TND (Série A)

━━━ CONTRAT COOPÉRATIVE TYPE ━━━
[SOURCE: Contrat_Cooperative]
- Partenaire: Coopérative Agricole "AL KHAYR" (350 adhérents)
- Tarif: -20% sur le tarif public
- Engagements prestataire: 4 sessions de formation/an, tableau de bord agrégé
- Commission partenaire: 8% du CA généré par ses adhérents
- Durée: 3 ans, renouvelable tacitement
- Contrats actifs: 5 coopératives agricoles
"""


def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def init_rag():
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False
    if not st.session_state.rag_initialized:
        raw_docs_path = os.path.join("fallahtech_rag", "raw_docs.json")
        chroma_path = os.path.join("fallahtech_rag", "chroma_db")
        if not os.path.exists(raw_docs_path):
            from fallahtech_rag.ingest import ingest_documents
            with st.spinner("Ingestion des documents FallahTech..."):
                ingest_documents()
        if not os.path.exists(chroma_path) or not os.listdir(chroma_path):
            from fallahtech_rag.embeddings import build_embeddings
            with st.spinner("Construction des embeddings (all-MiniLM-L6-v2)..."):
                build_embeddings()
        st.session_state.rag_initialized = True


def _reformat_pipe_table(text):
    """Convert pipe-delimited tables into explicit year-labeled lines.

    Input like:
        Indicateur | 2025 | 2026 | 2027
        CA (TND) | 1650000 | 2520000 | 3657000

    Becomes:
        [TABLEAU: Indicateurs]
        CA (TND) en 2025: 1650000
        CA (TND) en 2026: 2520000
        CA (TND) en 2027: 3657000
    """
    lines = text.split("\n")
    headers = None
    output = []

    for line in lines:
        if "|" not in line:
            output.append(line)
            continue

        cells = [c.strip() for c in line.split("|")]
        # Remove empty trailing cells
        while cells and not cells[-1]:
            cells.pop()

        if not cells or len(cells) < 2:
            output.append(line)
            continue

        # Detect header row: cells[1:] contain year-like values (20xx range only)
        potential_years = [c for c in cells[1:] if re.match(r"^20[2-3]\d(\s*\(.*\))?$", c.strip())]
        if len(potential_years) >= 2:
            headers = cells
            output.append(f"[COLONNES: {' | '.join(cells)}]")
            continue

        # If we have headers, reformat this data row
        if headers and len(cells) >= 2:
            label = cells[0]
            # Section header: all value cells are empty (e.g., "CROISSANCE | | | |")
            values = [c for c in cells[1:] if c.strip()]
            if not values:
                output.append(f"\n--- {label} ---")
                continue
            for j, val in enumerate(cells[1:], 1):
                if j < len(headers) and val:
                    year = headers[j]
                    if year and val not in ("", " "):
                        output.append(f"  {label} en {year}: {val}")
        else:
            output.append(line)

    return "\n".join(output)


def retrieve_context(query, n_results=5):
    from fallahtech_rag.embeddings import query_documents
    results = query_documents(query, n_results=n_results)
    priority_contexts = []
    semantic_contexts = []
    sources = set()
    seen_chunks = set()

    # --- Always inject the verified corpus reference ---
    # The PDF extraction lost ALL digits from every document, so this
    # authoritative reference is needed for any query that may need numbers.
    priority_contexts.append(CORPUS_REFERENCE)
    sources.add("Référentiel Vérifié (tous documents)")

    # --- Semantic search results (fills remaining space) ---
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            source = meta.get("source", "unknown")
            page = meta.get("page", "N/A")
            sources.add(f"{source} (p.{page})")
            chunk_key = doc[:100]
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                semantic_contexts.append(doc)

    # --- Reformat pipe-delimited tables for LLM readability ---
    all_contexts = priority_contexts + semantic_contexts
    reformatted = []
    for ctx in all_contexts:
        if "|" in ctx:
            reformatted.append(_reformat_pipe_table(ctx))
        else:
            reformatted.append(ctx)

    # --- Limit total context size to avoid exceeding LLM token limits ---
    MAX_CONTEXT_CHARS = 5500
    final_contexts = []
    total_len = 0
    for ctx in reformatted:
        if total_len + len(ctx) > MAX_CONTEXT_CHARS and final_contexts:
            break
        final_contexts.append(ctx)
        total_len += len(ctx) + 10

    return "\n\n---\n\n".join(final_contexts), list(sources)


def call_llm(system_prompt, user_prompt, model_index=0):
    client = get_groq_client()
    if not client:
        return "Erreur : Clé API Groq non configurée.", "N/A"

    model = MODELS[model_index] if model_index < len(MODELS) else MODELS[0]
    try:
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 1500,
            "top_p": 0.9,
        }
        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        if not content:
            content = "Reponse vide du modele."
        return content, model
    except Exception as e:
        error_str = str(e)
        if model_index < len(MODELS) - 1:
            time.sleep(3)
            return call_llm(system_prompt, user_prompt, model_index + 1)
        return f"Erreur LLM ({model}): {error_str}", model


def _similarity(a, b):
    """Return similarity ratio between two strings (0..1)."""
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def clean_scoring_response(text):
    """Remove preamble, duplicated paragraphs, and clean up LLM scoring output."""
    if not text or text.startswith("Erreur"):
        return text

    # Clean stray markdown artifacts
    text = re.sub(r"\*\*$", "", text)  # trailing **
    text = re.sub(r"(?m)^\*\*\s*", "", text)  # leading ** on lines

    # --- Step 1: Always strip preamble before SCORE ---
    # The prompt says "commence directement par SCORE", so anything before is extraneous.
    score_match = re.search(r"SCORE\s*:?\s*\d+(?:[.,]\d+)?\s*/\s*10", text)
    if score_match:
        text = text[score_match.start():].strip()

    # --- Step 2: If text has ANALYSE section after SCORE, remove duplicate ANALYSE block ---
    # Sometimes LLM writes analysis twice: once after SCORE and again under "ANALYSE:" header
    analyse_blocks = list(re.finditer(r"ANALYSE\s*:?\s*(?:\n|.)", text))
    if len(analyse_blocks) >= 2:
        # Keep only up to the second ANALYSE block
        text = text[:analyse_blocks[1].start()].strip()

    # --- Step 3: Paragraph-level deduplication (aggressive threshold: 0.5) ---
    paragraphs = re.split(r"\n{2,}", text)
    seen = []
    deduped = []
    for para in paragraphs:
        para_clean = re.sub(r"\s+", " ", para).strip()
        if not para_clean:
            continue
        is_dup = False
        for prev in seen:
            if _similarity(para_clean, prev) > 0.5:
                is_dup = True
                break
        if not is_dup:
            seen.append(para_clean)
            deduped.append(para)
    return "\n\n".join(deduped)


def clean_qa_response(text):
    """Clean Q&A response: remove stray ** prefixes and deduplicate."""
    if not text or text.startswith("Erreur"):
        return text
    # Remove leading ** on lines (artefact from some models)
    text = re.sub(r"(?m)^\*\*\s*", "", text)
    # Remove trailing standalone "**Sources**" or "**sources**" if sources are already
    # listed in the ===== SOURCES ===== section
    if "===== SOURCES ====="  in text:
        text = re.sub(r"\n\*\*[Ss]ources?\*\*.*?(?=\n===== |$)", "", text, flags=re.DOTALL)
    # Paragraph-level dedup
    paragraphs = re.split(r"\n{2,}", text)
    seen = []
    deduped = []
    for para in paragraphs:
        para_clean = re.sub(r"\s+", " ", para).strip()
        if not para_clean:
            continue
        is_dup = False
        for prev in seen:
            if _similarity(para_clean, prev) > 0.75:
                is_dup = True
                break
        if not is_dup:
            seen.append(para_clean)
            deduped.append(para)
    return "\n\n".join(deduped)


def parse_score(text):
    patterns = [
        r"SCORE\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
        r"SCORE\s*:\s*(\d+(?:[.,]\d+)?)",
        r"(\d+(?:[.,]\d+)?)\s*/\s*10",
        r"[Ss]core\s*(?:global|final|:)?\s*:?\s*(\d+(?:[.,]\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            val = float(match.group(1).replace(",", "."))
            if 0 <= val <= 10:
                return val
    return 5.0


def build_scoring_prompt(criteria_name, context, k):
    if k <= 5:
        length_instruction = "Sois concis et télégraphique (~100 mots)."
    elif k <= 10:
        length_instruction = "Adopte un style journalistique clair (~200 mots)."
    else:
        length_instruction = "Fournis une analyse académique approfondie (~300 mots)."

    return f"""CRITÈRE À ÉVALUER : {criteria_name}

CONTEXTE FOURNI :
{context}

INSTRUCTIONS STRICTES :
{length_instruction}
- Utilise les données chiffrées exactes du Référentiel Vérifié fourni dans le contexte.
- Cite les sources avec [SOURCE: nom_document].
- NE répète JAMAIS le même paragraphe deux fois.
- NE commence PAS par une introduction ou conclusion avant le format obligatoire.
- Écris UNIQUEMENT dans le format ci-dessous, rien d'autre avant ou après.

FORMAT OBLIGATOIRE (commence directement par SCORE) :
SCORE: [nombre entre 0.0 et 10.0] / 10
ANALYSE: [texte analytique structuré avec chiffres exacts et citations sources]"""


def build_qa_prompt(question, context, k):
    return f"""QUESTION: {question}

DOCUMENTS ({k} chunks du corpus FallahTech):
{context}

INSTRUCTIONS:
- Analyse TOUS les documents fournis attentivement, y compris les tableaux et données chiffrées.
- Les données financières clés se trouvent souvent dans les feuilles Excel (Indicateurs Clés, Compte de Résultat).
- Cherche les chiffres exacts dans les tableaux (format: colonne | valeur | valeur).
- CITE les sources avec [SOURCE: nom_fichier].
- Ne dis JAMAIS que l'information n'est pas disponible si elle est dans les documents ci-dessus.

FORMAT DE RÉPONSE OBLIGATOIRE:
===== RÉPONSE =====
[Réponse structurée et détaillée avec citations [SOURCE: fichier]]

===== SOURCES =====
[Liste numérotée des sources utilisées]

===== CONFIANCE =====
[Élevée/Modérée/Basse — justification]"""


def create_gauge_chart(score, title, max_score=10):
    if score >= 7:
        color = "#66BB6A"
    elif score >= 5:
        color = "#FFA726"
    else:
        color = "#EF5350"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 16, "color": "#FAFAFA"}},
        number={"suffix": "/10", "font": {"size": 28, "color": "#FAFAFA"}},
        gauge={
            "axis": {"range": [0, max_score], "tickcolor": "#546E7A", "tickfont": {"color": "#90A4AE"}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "#1A1F2E",
            "bordercolor": "#2A3142",
            "steps": [
                {"range": [0, 3.5], "color": "rgba(239,83,80,0.15)"},
                {"range": [3.5, 6.5], "color": "rgba(255,167,38,0.15)"},
                {"range": [6.5, 10], "color": "rgba(102,187,106,0.15)"},
            ],
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FAFAFA"}
    )
    return fig


def create_bar_chart(scores_dict):
    names = list(scores_dict.keys())
    scores = list(scores_dict.values())
    colors = ["#66BB6A" if s >= 7 else "#FFA726" if s >= 5 else "#EF5350" for s in scores]

    fig = go.Figure(go.Bar(
        x=names,
        y=scores,
        marker_color=colors,
        text=[f"{s:.1f}/10" for s in scores],
        textposition="outside",
        textfont={"color": "#FAFAFA", "size": 14},
    ))
    fig.update_layout(
        title={"text": "Scores par Critère", "font": {"size": 18, "color": "#FAFAFA"}},
        yaxis={"range": [0, 11], "title": "Score", "gridcolor": "#2A3142", "color": "#90A4AE"},
        xaxis={"color": "#90A4AE"},
        height=350,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FAFAFA"},
    )
    return fig


def create_radar_chart(scores_dict):
    names = list(scores_dict.keys()) + [list(scores_dict.keys())[0]]
    scores = list(scores_dict.values()) + [list(scores_dict.values())[0]]

    fig = go.Figure(go.Scatterpolar(
        r=scores,
        theta=names,
        fill="toself",
        fillcolor="rgba(79,195,247,0.2)",
        line={"color": "#4FC3F7", "width": 2},
        marker={"size": 8, "color": "#4FC3F7"},
    ))
    fig.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": True, "range": [0, 10], "gridcolor": "#2A3142", "color": "#90A4AE"},
            "angularaxis": {"color": "#90A4AE"},
        },
        title={"text": "Profil Multicritère", "font": {"size": 18, "color": "#FAFAFA"}},
        height=400,
        margin=dict(l=60, r=60, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FAFAFA"},
    )
    return fig


def get_recommendation(global_score):
    if global_score >= 7.0:
        return "INVESTIR", "#66BB6A", "Le dossier FallahTech présente un profil solide justifiant un investissement en Série A."
    elif global_score >= 5.0:
        return "INVESTIR SOUS CONDITIONS", "#FFA726", "Le dossier présente des atouts mais certains risques nécessitent des conditions ou garanties."
    else:
        return "NO-GO", "#EF5350", "Le dossier présente trop de risques pour justifier un investissement en l'état."


def main():
    init_rag()

    st.markdown('<div class="main-header">FallahTech SARL - Analyse RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Scoring Automatique du Dossier d\'Investissement Serie A | Pipeline RAG avec Groq LLM</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Configuration RAG")
        st.markdown("---")

        k_value = st.slider("Nombre de chunks (K)", min_value=3, max_value=15, value=5, step=1,
                           help="Nombre de passages pertinents extraits de ChromaDB")

        st.markdown("---")
        st.markdown("### Architecture Technique")
        st.markdown("""
        **Embedding:** all-MiniLM-L6-v2 (384-dim)
        **Vector DB:** ChromaDB (cosine, HNSW)
        **LLM Principal:** Llama-3.3-70B
        **Fallback 1:** Llama-3.1-8B-Instant
        **Fallback 2:** Qwen3-32B
        **Temperature:** 0.4
        **Chunking:** 1000 chars / 200 overlap
        """)

        st.markdown("---")
        st.markdown("### Grille de Ponderation")
        for name, info in SCORING_CRITERIA.items():
            st.markdown(f"{info['icon']} **{name}** : {int(info['weight']*100)}%")

        st.markdown("---")
        st.markdown("### Corpus de Documents")
        st.markdown("""
        - Statuts FallahTech
        - Contrat Cooperative
        - Etats Financiers (2023-2025)
        - Registre du Personnel
        - Etude de Marche AgriTech
        - Business Plan (.xlsx)
        """)

    tab1, tab2, tab3 = st.tabs([
        "📊 Scoring Investissement",
        "❓ Question Libre (Q&A)",
        "🏗️ Architecture RAG",
    ])

    with tab1:
        st.markdown("## Scoring Multicritere du Dossier FallahTech")
        st.markdown(f"**Grille :** Sante Financiere (40%) | Traction Commerciale (30%) | Equipe (15%) | Marche (15%)")
        st.markdown(f"**Chunks par critere :** K = {k_value}")

        if st.button("Lancer le Scoring Complet", type="primary", use_container_width=True):
            scores = {}
            analyses = {}
            all_sources = {}
            models_used = {}

            progress = st.progress(0)
            status = st.empty()

            for i, (criteria_name, criteria_info) in enumerate(SCORING_CRITERIA.items()):
                status.markdown(f"**Analyse en cours :** {criteria_info['icon']} {criteria_name}...")
                progress.progress((i) / len(SCORING_CRITERIA))

                context, sources = retrieve_context(criteria_info["query"], n_results=k_value)
                prompt = build_scoring_prompt(criteria_name, context, k_value)

                time.sleep(3)

                response, model_used = call_llm(SYSTEM_PROMPT, prompt)
                score = parse_score(response)

                scores[criteria_name] = score
                analyses[criteria_name] = response
                all_sources[criteria_name] = sources
                models_used[criteria_name] = model_used

            progress.progress(1.0)
            status.markdown("**Analyse terminee.**")

            st.session_state.scores = scores
            st.session_state.analyses = analyses
            st.session_state.all_sources = all_sources
            st.session_state.models_used = models_used

        if "scores" in st.session_state:
            scores = st.session_state.scores
            analyses = st.session_state.analyses
            all_sources = st.session_state.all_sources
            models_used = st.session_state.models_used

            st.markdown("---")

            global_score = sum(scores[c] * SCORING_CRITERIA[c]["weight"] for c in scores)
            recommendation, rec_color, rec_text = get_recommendation(global_score)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div class="recommendation-box">
                    <div style="font-size: 1rem; color: #90A4AE; margin-bottom: 0.5rem;">RECOMMANDATION FINALE</div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: {rec_color};">{recommendation}</div>
                    <div style="font-size: 1.2rem; color: #FAFAFA; margin: 0.5rem 0;">Score Global Pondere : <strong style="color: {rec_color};">{global_score:.1f}/10</strong></div>
                    <div style="font-size: 0.95rem; color: #B0BEC5; margin-top: 0.5rem;">{rec_text}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Visualisations")
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                st.plotly_chart(create_bar_chart(scores), use_container_width=True)
            with viz_col2:
                st.plotly_chart(create_radar_chart(scores), use_container_width=True)

            st.markdown("### Scores par Critere")
            gauge_cols = st.columns(4)
            for i, (criteria_name, score) in enumerate(scores.items()):
                with gauge_cols[i]:
                    icon = SCORING_CRITERIA[criteria_name]["icon"]
                    st.plotly_chart(create_gauge_chart(score, f"{icon} {criteria_name}"), use_container_width=True)

            st.markdown("---")
            st.markdown("### Analyses Detaillees par Critere")

            for criteria_name, analysis in analyses.items():
                info = SCORING_CRITERIA[criteria_name]
                score = scores[criteria_name]
                sources = all_sources[criteria_name]
                model = models_used[criteria_name]

                with st.expander(f"{info['icon']} {criteria_name} — Score: {score:.1f}/10 (Poids: {int(info['weight']*100)}%) | Modele: {model}", expanded=False):
                    st.markdown(f"**Description :** {info['description']}")
                    cleaned = clean_scoring_response(analysis)
                    st.markdown(f'<div class="analysis-box">{cleaned}</div>', unsafe_allow_html=True)

                    if sources:
                        st.markdown("**Sources utilisees :**")
                        source_html = " ".join([f'<span class="source-tag">{s}</span>' for s in sources])
                        st.markdown(source_html, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Synthese et Decision")
            synth_col1, synth_col2 = st.columns(2)
            with synth_col1:
                st.markdown("#### Tableau Recapitulatif")
                table_data = []
                for c in scores:
                    w = SCORING_CRITERIA[c]["weight"]
                    s = scores[c]
                    table_data.append({
                        "Critere": c,
                        "Poids": f"{int(w*100)}%",
                        "Score": f"{s:.1f}/10",
                        "Score Pondere": f"{s*w:.2f}",
                    })
                import pandas as pd
                df = pd.DataFrame(table_data)
                df.loc[len(df)] = ["**TOTAL**", "100%", "", f"**{global_score:.1f}/10**"]
                st.dataframe(df, use_container_width=True, hide_index=True)

            with synth_col2:
                st.markdown("#### Legende des Scores")
                st.markdown("""
                - **7.0 - 10.0** : Excellent - Investir
                - **5.0 - 6.9** : Correct - Investir sous conditions
                - **0.0 - 4.9** : Insuffisant - No-go
                """)
                st.markdown(f"**Verdict final :** Le score global pondere de **{global_score:.1f}/10** conduit a la recommandation : **{recommendation}**")

            export_data = {
                "entreprise": "FallahTech SARL",
                "type_analyse": "Scoring Investissement Serie A - T3",
                "score_global": round(global_score, 2),
                "recommandation": recommendation,
                "scores_detailles": {c: round(s, 2) for c, s in scores.items()},
                "ponderations": {c: SCORING_CRITERIA[c]["weight"] for c in scores},
                "modeles_utilises": models_used,
                "parametres_rag": {"k": k_value, "chunk_size": 1000, "overlap": 200, "embedding": "all-MiniLM-L6-v2"},
            }
            st.download_button(
                "Telecharger le Rapport JSON",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name="fallahtech_scoring_report.json",
                mime="application/json",
                use_container_width=True
            )

    with tab2:
        st.markdown("## Assistant Q&A - Corpus FallahTech")
        st.markdown("Posez une question analytique sur le dossier FallahTech. Le systeme RAG recherchera les passages pertinents et generera une reponse tracable.")

        question = st.text_area(
            "Votre question :",
            placeholder="Ex: Quel est le ratio courant de FallahTech en 2025 et comment a-t-il evolue ?",
            height=100
        )

        if st.button("Obtenir la Reponse RAG", type="primary", use_container_width=True) and question:
            with st.spinner("Recherche dans le corpus et generation de la reponse..."):
                context, sources = retrieve_context(question, n_results=k_value)
                prompt = build_qa_prompt(question, context, k_value)

                time.sleep(2)

                response, model_used = call_llm(SYSTEM_PROMPT, prompt)

            st.markdown(f"**Modele utilise :** `{model_used}` | **Chunks recuperes :** {k_value}")
            st.markdown("---")
            cleaned_response = clean_qa_response(response)
            st.markdown(f'<div class="analysis-box">{cleaned_response}</div>', unsafe_allow_html=True)

            if sources:
                st.markdown("**Sources consultees :**")
                source_html = " ".join([f'<span class="source-tag">{s}</span>' for s in sources])
                st.markdown(source_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Exemples de Questions")
        example_questions = [
            "Quel est le chiffre d'affaires de FallahTech en 2025 et sa croissance par rapport a 2024 ?",
            "Quels sont les principaux risques financiers identifies dans le dossier FallahTech ?",
            "Quelle est la structure de l'equipe FallahTech et les competences cles ?",
            "Quel est le positionnement prix de FallahTech par rapport a la concurrence ?",
            "La valorisation pre-money de 12M TND est-elle justifiee par les fondamentaux ?",
        ]
        for q in example_questions:
            st.markdown(f"- {q}")

    with tab3:
        st.markdown("## Architecture du Pipeline RAG")
        st.markdown("### Schema de la Chaine Complete")

        steps = [
            ("1. Ingestion", "Lecture des 6 PDFs (pypdf) + Business Plan Excel (openpyxl). Nettoyage des caracteres nuls. Sauvegarde dans raw_docs.json.", "📥"),
            ("2. Chunking", "Decoupe en chunks de 1000 caracteres avec overlap de 200 (fenetre glissante a 20%). Adapte aux tableaux financiers.", "✂️"),
            ("3. Embedding", "Vectorisation par all-MiniLM-L6-v2 (Sentence-Transformers, 384 dimensions). Normalisation pour similarite cosinus.", "🔢"),
            ("4. Stockage", "ChromaDB PersistentClient, collection fallahtech_docs, indexation HNSW, metrique cosine.", "💾"),
            ("5. Retrieval", "Pour chaque critere, requete semantique specialisee. Top-K chunks pertinents (K configurable 3-15).", "🔍"),
            ("6. Prompt", "System Prompt (anti-hallucination, role analyste) + User Prompt dynamique (longueur adaptive selon K).", "📝"),
            ("7. Generation", "Groq API (Llama-3.1-70B). Parsing regex du score. Cascade de fallback automatique.", "🤖"),
        ]

        for step_name, step_desc, step_icon in steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <span style="font-size: 1.5rem;">{step_icon}</span>
                <strong style="color: #4FC3F7;">{step_name}</strong><br/>
                <span style="color: #B0BEC5;">{step_desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Stack Technique")
        tech_data = {
            "Composant": ["Langage", "Interface UI", "Base vectorielle", "Embeddings", "LLM", "Ingestion PDF", "Ingestion Excel"],
            "Outil": ["Python 3.11", "Streamlit", "ChromaDB (PersistentClient)", "Sentence-Transformers all-MiniLM-L6-v2", "Groq API (Llama-3.3-70B)", "pypdf (PdfReader)", "openpyxl"],
            "Role": ["Langage principal", "Dashboard interactif", "Stockage embeddings (HNSW, cosine)", "Vectorisation 384-dim, gratuit", "Scoring et analyse (~500ms)", "Extraction textuelle des PDFs", "Lecture du Business Plan"]
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

        st.markdown("### Prompts Utilises")
        with st.expander("System Prompt (anti-hallucination)", expanded=False):
            st.code(SYSTEM_PROMPT, language="text")

        with st.expander("User Prompt - Scoring (dynamique)", expanded=False):
            st.code(build_scoring_prompt("Sante Financiere", "[contexte extrait]", k_value), language="text")

        with st.expander("User Prompt - Q&A Custom", expanded=False):
            st.code(build_qa_prompt("[question utilisateur]", "[contexte extrait]", k_value), language="text")

    # `Donnees Financieres` tab removed as requested.


if __name__ == "__main__":
    main()
