# bm25_retrieval.py
# Baseline truy hồi sử dụng BM25 (Best Matching 25)
# ========================================================
# BM25 là phương pháp truy hồi dựa trên thống kê từ,
# sử dụng TF (term frequency) có chuẩn hóa độ dài văn bản
# và IDF (inverse document frequency).
#
# Công thức BM25:
# score(q, d) = Σ IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1 * (1 - b + b * |d|/avgdl))
#
# Trong đó:
# - k1: tham số điều chỉnh ảnh hưởng của term frequency (thường = 1.2 - 2.0)
# - b: tham số điều chỉnh ảnh hưởng của độ dài văn bản (thường = 0.75)
# - |d|: độ dài văn bản d (số tokens)
# - avgdl: độ dài trung bình của tất cả văn bản trong corpus

import json
import math
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.vietnamese_preprocessor import preprocess, preprocess_corpus


class BM25Retriever:
    """BM25 Retriever cho truy hồi văn bản tiếng Việt.
    
    Sử dụng thuật toán Okapi BM25 để xếp hạng các passages
    theo mức độ phù hợp với query.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75,
                 use_underthesea: bool = False,
                 remove_stopwords: bool = True):
        """
        Args:
            k1: Tham số TF saturation (1.2 - 2.0)
            b: Tham số document length normalization (0 - 1)
            use_underthesea: Có dùng underthesea tách từ không
            remove_stopwords: Có loại bỏ stopwords không
        """
        self.k1 = k1
        self.b = b
        self.use_underthesea = use_underthesea
        self.remove_stopwords = remove_stopwords

        # Sẽ được khởi tạo khi gọi fit()
        self.corpus_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        self.df: Dict[str, int] = {}       # document frequency
        self.idf: Dict[str, float] = {}     # inverse document frequency
        self.passage_ids: List[str] = []
        self.passages: List[dict] = []

    def fit(self, passages: List[dict]):
        """Index toàn bộ passages.
        
        Args:
            passages: List of dicts với keys: passage_id, text, clause_ref
        """
        print(f"🔧 BM25: Indexing {len(passages)} passages...")
        start = time.time()

        self.passages = passages
        self.passage_ids = [p["passage_id"] for p in passages]
        texts = [p["text"] for p in passages]

        # Tokenize corpus
        self.corpus_tokens = preprocess_corpus(
            texts,
            use_underthesea=self.use_underthesea,
            remove_sw=self.remove_stopwords
        )

        self.n_docs = len(self.corpus_tokens)
        self.doc_lens = [len(doc) for doc in self.corpus_tokens]
        self.avgdl = sum(self.doc_lens) / max(self.n_docs, 1)

        # Tính Document Frequency (DF)
        self.df = {}
        for doc_tokens in self.corpus_tokens:
            unique_terms = set(doc_tokens)
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1

        # Tính IDF (Robertson-Spärck Jones formula)
        self.idf = {}
        for term, df in self.df.items():
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = math.log(
                (self.n_docs - df + 0.5) / (df + 0.5) + 1.0
            )

        elapsed = time.time() - start
        print(f"✅ BM25: Indexed in {elapsed:.2f}s | "
              f"vocab={len(self.df)} | avgdl={self.avgdl:.1f}")

    def _score_document(self, query_tokens: List[str],
                        doc_idx: int) -> float:
        """Tính BM25 score cho 1 cặp (query, document)."""
        doc_tokens = self.corpus_tokens[doc_idx]
        doc_len = self.doc_lens[doc_idx]

        # Term frequency trong document
        tf_doc = Counter(doc_tokens)

        score = 0.0
        for qt in query_tokens:
            if qt not in self.idf:
                continue
            tf = tf_doc.get(qt, 0)
            idf = self.idf[qt]

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self.avgdl
            )
            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """Truy hồi top-K passages phù hợp nhất với query.
        
        Args:
            query: Câu truy vấn
            top_k: Số lượng kết quả trả về
            
        Returns:
            List of (passage_id, score, passage_dict)
        """
        query_tokens = preprocess(
            query,
            use_underthesea=self.use_underthesea,
            remove_sw=self.remove_stopwords
        )

        # Tính score cho tất cả documents
        scores = np.array([
            self._score_document(query_tokens, i)
            for i in range(self.n_docs)
        ])

        # Top-K indices
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
    """Demo sử dụng BM25 Retriever tương tác."""
    PASSAGES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "passages.json")

    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        passages = json.load(f)
    if isinstance(passages, dict) and "data" in passages:
        passages = passages["data"]

    # Khởi tạo và index
    retriever = BM25Retriever(k1=1.5, b=0.75, use_underthesea=False)
    retriever.fit(passages)

    # Truy vấn tương tác
    print("\n" + "="*60)
    print("🔍 BM25 Retriever — Truy vấn tương tác")
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
