# tfidf_retrieval.py
# Baseline truy hồi sử dụng TF-IDF + Cosine Similarity
# ========================================================
# TF-IDF (Term Frequency - Inverse Document Frequency) là phương pháp
# biểu diễn văn bản thành vector thưa (sparse vector), trong đó mỗi
# chiều tương ứng với 1 từ trong vocabulary.
#
# TF-IDF(t, d) = TF(t, d) × IDF(t)
# - TF(t, d) = số lần xuất hiện của từ t trong document d
# - IDF(t) = log(N / df(t))
#
# Cosine Similarity giữa query vector và document vector cho điểm xếp hạng.

import json
import time
import numpy as np
from typing import List, Dict, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.vietnamese_preprocessor import preprocess, preprocess_corpus


class TFIDFRetriever:
    """TF-IDF Retriever cho truy hồi văn bản tiếng Việt.
    
    Xây dựng TF-IDF matrix từ corpus passages, sau đó dùng
    cosine similarity để xếp hạng passages theo query.
    """

    def __init__(self, use_underthesea: bool = False,
                 remove_stopwords: bool = True,
                 sublinear_tf: bool = True,
                 smooth_idf: bool = True):
        """
        Args:
            use_underthesea: Có dùng underthesea tách từ không
            remove_stopwords: Có loại bỏ stopwords không
            sublinear_tf: Dùng 1 + log(tf) thay vì tf thô (tốt hơn cho văn bản dài)
            smooth_idf: Có smooth IDF hay không (tránh division by zero)
        """
        self.use_underthesea = use_underthesea
        self.remove_stopwords = remove_stopwords
        self.sublinear_tf = sublinear_tf
        self.smooth_idf = smooth_idf

        # Sẽ được khởi tạo khi gọi fit()
        self.vocab: Dict[str, int] = {}     # term → index
        self.idf: np.ndarray = None         # IDF values
        self.tfidf_matrix: np.ndarray = None  # (n_docs, vocab_size)
        self.passage_ids: List[str] = []
        self.passages: List[dict] = []

    def fit(self, passages: List[dict]):
        """Xây dựng TF-IDF matrix từ corpus.
        
        Args:
            passages: List of dicts với keys: passage_id, text, clause_ref
        """
        print(f"🔧 TF-IDF: Building matrix for {len(passages)} passages...")
        start = time.time()

        self.passages = passages
        self.passage_ids = [p["passage_id"] for p in passages]
        texts = [p["text"] for p in passages]

        # Tokenize corpus
        corpus_tokens = preprocess_corpus(
            texts,
            use_underthesea=self.use_underthesea,
            remove_sw=self.remove_stopwords
        )

        n_docs = len(corpus_tokens)

        # 1. Xây dựng vocabulary
        self.vocab = {}
        for doc_tokens in corpus_tokens:
            for token in doc_tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        vocab_size = len(self.vocab)
        print(f"   Vocabulary size: {vocab_size}")

        # 2. Tính Term Frequency matrix (n_docs × vocab_size)
        tf_matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)
        for doc_idx, doc_tokens in enumerate(corpus_tokens):
            for token in doc_tokens:
                tid = self.vocab[token]
                tf_matrix[doc_idx, tid] += 1

        # Sublinear TF: 1 + log(tf) thay vì tf
        if self.sublinear_tf:
            mask = tf_matrix > 0
            tf_matrix[mask] = 1 + np.log(tf_matrix[mask])

        # 3. Tính IDF
        df = np.sum(tf_matrix > 0, axis=0).astype(np.float64)  # document frequency
        if self.smooth_idf:
            # smooth IDF: log((1 + N) / (1 + df)) + 1
            self.idf = np.log((1 + n_docs) / (1 + df)) + 1
        else:
            # standard IDF: log(N / df)
            self.idf = np.log(n_docs / np.maximum(df, 1e-10))

        # 4. TF-IDF = TF × IDF
        self.tfidf_matrix = tf_matrix * self.idf[np.newaxis, :]

        # 5. L2 Normalize mỗi document vector (để cosine similarity = dot product)
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.tfidf_matrix = self.tfidf_matrix / norms

        elapsed = time.time() - start
        print(f"✅ TF-IDF: Built in {elapsed:.2f}s | "
              f"matrix shape={self.tfidf_matrix.shape}")

    def _query_to_vector(self, query: str) -> np.ndarray:
        """Chuyển query thành TF-IDF vector."""
        query_tokens = preprocess(
            query,
            use_underthesea=self.use_underthesea,
            remove_sw=self.remove_stopwords
        )

        vocab_size = len(self.vocab)
        q_vec = np.zeros(vocab_size, dtype=np.float64)

        for token in query_tokens:
            if token in self.vocab:
                tid = self.vocab[token]
                q_vec[tid] += 1

        # Sublinear TF
        if self.sublinear_tf:
            mask = q_vec > 0
            q_vec[mask] = 1 + np.log(q_vec[mask])

        # Apply IDF
        q_vec = q_vec * self.idf

        # L2 normalize
        norm = np.linalg.norm(q_vec)
        if norm > 1e-10:
            q_vec = q_vec / norm

        return q_vec

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """Truy hồi top-K passages phù hợp nhất với query.
        
        Args:
            query: Câu truy vấn
            top_k: Số lượng kết quả trả về
            
        Returns:
            List of (passage_id, score, passage_dict)
        """
        q_vec = self._query_to_vector(query)

        # Cosine similarity (= dot product vì đã normalize)
        scores = self.tfidf_matrix @ q_vec

        # Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            pid = self.passage_ids[idx]
            results.append((pid, scores[idx], self.passages[idx]))

        return results

    def batch_retrieve(self, queries: List[str],
                       top_k: int = 20) -> List[List[Tuple[str, float]]]:
        """Truy hồi batch queries. Trả về list of (passage_id, score) cho mỗi query."""
        all_results = []
        for q in queries:
            results = self.retrieve(q, top_k=top_k)
            all_results.append([(pid, score) for pid, score, _ in results])
        return all_results


# ====== STANDALONE USAGE ======
def main():
    """Demo sử dụng TF-IDF Retriever tương tác."""
    PASSAGES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "passages.json")

    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        passages = json.load(f)
    if isinstance(passages, dict) and "data" in passages:
        passages = passages["data"]

    # Khởi tạo và index
    retriever = TFIDFRetriever(use_underthesea=False, sublinear_tf=True)
    retriever.fit(passages)

    # Truy vấn tương tác
    print("\n" + "="*60)
    print("🔍 TF-IDF Retriever — Truy vấn tương tác")
    print("="*60)
    while True:
        q = input("\nQuery (enter để thoát): ").strip()
        if not q:
            break
        results = retriever.retrieve(q, top_k=5)
        for rank, (pid, score, passage) in enumerate(results, 1):
            print(f"\n  Rank {rank} — score={score:.4f} — {pid}")
            print(f"  {passage.get('clause_ref', '')}")
            print(f"  {passage.get('text', '')[:200]}...")


if __name__ == "__main__":
    main()
