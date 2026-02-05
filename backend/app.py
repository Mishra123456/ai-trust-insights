from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

# -----------------------------
# ML / NLP
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict

# Safe NLTK init (no repeated downloads)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="TrustScope – Human–AI Trust Intelligence")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# NLP Feature Extraction
# -----------------------------
def extract_nlp_features(df):
    sia = SentimentIntensityAnalyzer()

    df["sentiment"] = df["confidence_note"].fillna("").apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    df["skepticism_flag"] = df["confidence_note"].str.contains(
        "wrong|missed|override|manual|human|uncertain|mismatch|bias|doubt",
        case=False,
        na=False,
    ).astype(int)

    return df

# -----------------------------
# ML Model
# -----------------------------
def train_trust_model(df):
    X = df[["sentiment", "skepticism_flag"]]
    y = df["override"]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def explain_model(pipeline):
    coef = pipeline.named_steps["model"].coef_[0]
    return {
        "sentiment_weight": float(coef[0]),
        "skepticism_weight": float(coef[1]),
    }

# -----------------------------
# Trust Metrics
# -----------------------------
def calculate_metrics(df):
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W").astype(str)

    return (
        df.groupby("week")
        .agg(
            override_rate=("override", "mean"),
            trust_score=("override", lambda x: 1 - x.mean()),
            avg_sentiment=("sentiment", "mean"),
            total_cases=("override", "count"),
        )
        .reset_index()
    )

# -----------------------------
# Advanced Trust Intelligence
# -----------------------------
def trust_volatility(metrics):
    return float(metrics["trust_score"].std() or 0.0)


def trust_decay_rate(metrics):
    if len(metrics) < 2:
        return 0.0
    x = np.arange(len(metrics))
    y = metrics["trust_score"].values
    return float(np.polyfit(x, y, 1)[0])


def human_ai_alignment_index(metrics):
    return float(
        0.5 * metrics["trust_score"].mean()
        + 0.3 * (1 - metrics["override_rate"].mean())
        + 0.2 * ((metrics["avg_sentiment"].mean() + 1) / 2)
    )


def system_health_label(score):
    if score > 0.7:
        return "HEALTHY"
    if score > 0.45:
        return "AT RISK"
    return "CRITICAL"


def intervention_priority(metrics):
    rows = []
    for _, r in metrics.iterrows():
        score = (
            (1 - r["trust_score"]) * 0.5
            + r["override_rate"] * 0.3
            + abs(r["avg_sentiment"]) * 0.2
        )
        rows.append({"week": r["week"], "priority_score": round(score, 2)})
    return sorted(rows, key=lambda x: x["priority_score"], reverse=True)


def counterfactual_trust_gain(weights):
    return round(abs(weights["skepticism_weight"]) * 0.25, 3)

# -----------------------------
# RAG-STYLE THEME ANALYSIS (STABLE)
# -----------------------------
# -----------------------------
# RAG-STYLE THEME ANALYSIS (SEMANTIC)
# -----------------------------
from sentence_transformers import SentenceTransformer
import faiss

class TrustRAG:
    _model = None
    _index = None
    _categories = []
    
    @classmethod
    def initialize(cls):
        if cls._model is not None:
            return
            
        # Load lightweight efficient model
        cls._model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define semantic prototypes for categories
        # "RAG" strategy: Retrieve the closest category concept
        prototypes = {
            "Model Inaccuracy": [
                "wrong prediction", "incorrect outcome", "error in model", 
                "false positive", "hallucination", "bad output"
            ],
            "Context Failure": [
                "missed context", "didn't understand nuance", "sarcasm", 
                "ambiguous meaning", "domain mismatch", "out of distribution"
            ],
            "Human Preference": [
                "manual override", "human judgment", "stylistic choice", 
                "personal preference", "better phrasing", "editorial decision"
            ],
            "General Skepticism": [
                "uncertain", "doubt", "not sure", "verify later", "skeptical"
            ]
        }
        
        cls._categories = []
        vectors = []
        
        for category, phrases in prototypes.items():
            for p in phrases:
                cls._categories.append(category)
                vectors.append(cls._model.encode(p))
        
        # Build FAISS index for fast retrieval
        dimension = vectors[0].shape[0]
        cls._index = faiss.IndexFlatL2(dimension)
        cls._index.add(np.array(vectors))

    def __init__(self):
        if self._model is None:
            self.initialize()

    def classify(self, text):
        if not text or not isinstance(text, str):
            return "General Skepticism"
            
        vector = self._model.encode([text])
        # Find closest prototype
        D, I = self._index.search(vector, 1)
        closest_index = I[0][0]
        return self._categories[closest_index]

    def build(self, notes):
        buckets = defaultdict(list)
        for n in notes:
            buckets[self.classify(n)].append(n)

        return [
            {"theme": k, "count": len(v), "example": v[0]}
            for k, v in sorted(buckets.items(), key=lambda x: len(x[1]), reverse=True)
            if v
        ]

# Initialize model at module load to prevent per-request latency
try:
    TrustRAG.initialize()
except Exception as e:
    print(f"Warning: ML Model failed to load: {e}")


# -----------------------------
# Executive Summary
# -----------------------------
def executive_summary(metrics, weights, rag, haai):
    summary = []

    if metrics["trust_score"].min() < 0.5:
        summary.append("Trust declined during periods of elevated overrides.")

    summary.append(f"Primary trust failure driver: {rag[0]['theme']}.")
    strongest = max(weights, key=lambda k: abs(weights[k]))
    summary.append(f"ML attribution identifies '{strongest}' as the dominant predictor.")
    summary.append(f"Overall system health classified as {system_health_label(haai)}.")

    return " ".join(summary)

# -----------------------------
# MAIN ENDPOINT
# -----------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    required = {"date", "model_decision", "human_decision", "confidence_note"}
    if not required.issubset(df.columns):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain columns: {required}",
        )

    df["override"] = df["model_decision"] != df["human_decision"]
    df = extract_nlp_features(df)

    pipeline = train_trust_model(df)
    weights = explain_model(pipeline)

    df["override_risk"] = pipeline.predict_proba(
        df[["sentiment", "skepticism_flag"]]
    )[:, 1]

    df["risk_tier"] = pd.cut(
        df["override_risk"], [0, 0.4, 0.7, 1], labels=["LOW", "MEDIUM", "HIGH"]
    )

    metrics = calculate_metrics(df)
    rag = TrustRAG().build(df["confidence_note"].dropna().tolist())
    haai = human_ai_alignment_index(metrics)

    return {
        "metrics": metrics.to_dict("records"),
        "ml_weights": weights,
        "top_risks": df.sort_values("override_risk", ascending=False)
        .head(5)[["confidence_note", "override_risk", "risk_tier"]]
        .to_dict("records"),
        "rag_explanations": rag,
        "executive_summary": executive_summary(metrics, weights, rag, haai),
        "advanced_analysis": {
            "trust_volatility": trust_volatility(metrics),
            "trust_decay_rate": trust_decay_rate(metrics),
            "human_ai_alignment_index": haai,
            "system_health": system_health_label(haai),
            "intervention_priority": intervention_priority(metrics),
            "counterfactual_trust_gain": counterfactual_trust_gain(weights),
        },
    }
