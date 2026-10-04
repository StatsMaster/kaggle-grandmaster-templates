"""Small RAG retriever: embed -> index -> retrieve -> prompt.

The other half of the LLM Science Exam win: ground the model in retrieved
passages instead of hoping it memorized the answer.

Usage:
    python templates/nlp/rag_retriever.py
TODO: point CORPUS at your documents (one text per line or a CSV column).
"""
from pathlib import Path

import numpy as np

# TODO: your corpus — one document per line.
CORPUS_PATH = Path("data/corpus.txt")
TOP_K = 5


class Retriever:
    def __init__(self, docs):
        self.docs = docs
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer("all-MiniLM-L6-v2")
            self._emb = self._st.encode(docs, normalize_embeddings=True)
            self._dense = True
            print("[rag] dense embeddings (sentence-transformers)")
        except ImportError:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self._vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
            self._emb = self._vec.fit_transform(docs)
            self._cos = cosine_similarity
            self._dense = False
            print("[rag] TF-IDF fallback (install sentence-transformers for dense)")

    def retrieve(self, query, k=TOP_K):
        if self._dense:
            q = self._st.encode([query], normalize_embeddings=True)
            scores = q @ self._emb.T
        else:
            q = self._vec.transform([query])
            scores = self._cos(q, self._emb)
        idx = np.argsort(scores[0])[::-1][:k]
        return [(self.docs[i], float(scores[0][i])) for i in idx]


PROMPT_TEMPLATE = """Answer the question using ONLY the context below. If the answer
is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question, passages):
    context = "\n---\n".join(p for p, _ in passages)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def main():
    # TODO: replace with your corpus + questions.
    docs = (CORPUS_PATH.read_text().splitlines()
            if CORPUS_PATH.exists()
            else ["Paris is the capital of France.", "LoRA trains low-rank adapters."])
    retriever = Retriever(docs)
    question = "What is the capital of France?"  # TODO: your question
    passages = retriever.retrieve(question)
    prompt = build_prompt(question, passages)
    print(prompt)
    # TODO: send `prompt` to your (LoRA-tuned) model. Winners ensemble the
    # generator across seeds/prompts — see templates/blending/.


if __name__ == "__main__":
    main()
