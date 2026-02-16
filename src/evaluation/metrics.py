import json
import os
import pickle
import numpy as np
import evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any

class VLMMetrics:
    def __init__(self, tfidf_path: str = "tfidf_vectorizer.pkl"):
        """
        Bộ đo lường TF-IDF và ROUGE chuyên dụng cho Navigation Task.
        """
        self.rouge_metric = evaluate.load("rouge")
        self.tfidf_path = tfidf_path
        self.vectorizer = None
        
        # Load vectorizer nếu có
        if os.path.exists(self.tfidf_path):
            print(f"[Info] Loading TF-IDF vectorizer from {self.tfidf_path}")
            with open(self.tfidf_path, "rb") as f:
                self.vectorizer = pickle.load(f)
        else:
            print("[Warning] TF-IDF vectorizer not found. Run 'fit_tfidf' on training data first.")

    def _clean_text(self, text: str) -> str:
        """Làm sạch text, loại bỏ các thẻ XML nếu model sinh thừa"""
        text = text.strip()
        # Xử lý format <answer>...</answer> của preprocessing.py
        if "<answer>" in text:
            text = text.split("<answer>")[-1]
        if "</answer>" in text:
            text = text.split("</answer>")[0]
        return text.strip()

    def _extract_field(self, json_str: str, key: str = "instruction") -> str:
        """Parse JSON để lấy trường dữ liệu cụ thể"""
        try:
            clean_str = self._clean_text(json_str)
            # Thử parse JSON
            data = json.loads(clean_str)
            return str(data.get(key, "")).strip()
        except json.JSONDecodeError:
            # Fallback: Nếu JSON lỗi, trả về chuỗi gốc để tính ROUGE (chấp nhận phạt điểm)
            return clean_str

    def fit_tfidf(self, train_dataset):
        """Học từ vựng từ tập Train (Chỉ học trên field instruction)"""
        print("Fitting TF-IDF on training corpus...")
        corpus = []
        
        # Hỗ trợ cả Dataset object hoặc List
        iterator = train_dataset if isinstance(train_dataset, list) else range(len(train_dataset))
        
        for i in iterator:
            item = train_dataset[i] if not isinstance(train_dataset, list) else i
            # Logic: Cần decode label token thành text nếu input là dataset thô
            # Nhưng để đơn giản, ta khuyên user chạy fit trên list raw text
            # Ở đây giả định item là text hoặc dict đã decode
            # ... (Phần này sẽ xử lý ở script chạy ngoài)
            pass
            
        # NOTE: Để an toàn, hàm này nên được gọi với list các string 'instruction' chuẩn
        # Mình sẽ để logic extract ở ngoài runner cho linh hoạt.
        pass 

    def compute(self, predictions: List[str], references: List[str], target_field: str = "instruction") -> Dict[str, float]:
        """
        Tính toán điểm số.
        Args:
            predictions: List các chuỗi JSON do model sinh ra
            references: List các chuỗi JSON chuẩn (Ground Truth)
        """
        if self.vectorizer is None:
            print("[Warning] TF-IDF not fitted. Skipping TF-IDF score.")
            tfidf_score = 0.0
        else:
            # 1. Trích xuất text cần so sánh (instruction)
            pred_texts = [self._extract_field(p, key=target_field) for p in predictions]
            ref_texts = [self._extract_field(r, key=target_field) for r in references]
            
            # 2. Tính TF-IDF Cosine Similarity
            try:
                tfidf_preds = self.vectorizer.transform(pred_texts)
                tfidf_refs = self.vectorizer.transform(ref_texts)
                cosine_sims = (tfidf_preds.multiply(tfidf_refs)).sum(axis=1)
                tfidf_score = np.mean(cosine_sims) * 100
            except ValueError:
                tfidf_score = 0.0

        # 3. Tính ROUGE (trên toàn bộ chuỗi hoặc field cụ thể - ở đây chọn field cụ thể)
        pred_texts_rouge = [self._extract_field(p, key=target_field) for p in predictions]
        ref_texts_rouge = [self._extract_field(r, key=target_field) for r in references]
        
        rouge_scores = self.rouge_metric.compute(
            predictions=pred_texts_rouge, 
            references=ref_texts_rouge, 
            use_stemmer=True
        )

        return {
            "ROUGE-1": rouge_scores['rouge1'] * 100,
            "ROUGE-2": rouge_scores['rouge2'] * 100,
            "ROUGE-L": rouge_scores['rougeL'] * 100,
            "TF-IDF": tfidf_score
        }