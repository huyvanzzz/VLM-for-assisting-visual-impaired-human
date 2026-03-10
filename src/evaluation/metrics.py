import json
import evaluate
import numpy as np
from typing import List, Dict, Any

class VLMMetrics:
    def __init__(self, tfidf_path: str = "tfidf_vectorizer.pkl"):
        """
        [Đã chỉnh sửa] Chỉ load BERTScore để chạy nhanh. 
        Giữ nguyên tham số tfidf_path để không làm lỗi pipeline cũ.
        """
        print("[Info] Loading BERTScore metric...")
        self.bertscore_metric = evaluate.load("bertscore")
        self.tfidf_path = tfidf_path 
        
        # Tự động nhận diện GPU để ép BERTScore chạy nhanh hơn
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Info] BERTScore using device: {self.device.upper()}")

    def _clean_text(self, text: str) -> str:
        """Làm sạch text, loại bỏ thẻ XML"""
        text = text.strip()
        if "<answer>" in text:
            text = text.split("<answer>")[-1]
        if "</answer>" in text:
            text = text.split("</answer>")[0]
        return text.strip()

    def _extract_field(self, json_str: str, key: str = "instruction") -> str:
        """Parse JSON để lấy trường dữ liệu"""
        try:
            clean_str = self._clean_text(json_str)
            data = json.loads(clean_str)
            return str(data.get(key, "")).strip()
        except json.JSONDecodeError:
            return ""

    def fit_tfidf(self, corpus: List[str]):
        """
        Hàm giả (Dummy function): Giữ lại tên hàm để nhỡ các file khác 
        có gọi metrics.fit_tfidf() thì code không bị crash.
        """
        print("[Info] Skipped TF-IDF fitting (Focusing only on BERTScore).")
        pass

    def compute(self, predictions: List[str], references: List[str], target_field: str = "instruction") -> Dict[str, float]:
        """
        Giữ nguyên cách gọi hàm cũ: metrics.compute(preds, refs)
        Chỉ tính và trả về BERTScore.
        """
        pred_texts = [self._extract_field(p, key=target_field) for p in predictions]
        ref_texts = [self._extract_field(r, key=target_field) for r in references]

        try:
            # Chú ý: Đang để mặc định lang="en". 
            # Nếu data của bạn là tiếng Việt, hãy đổi thành lang="vi"
            results = self.bertscore_metric.compute(
                predictions=pred_texts,
                references=ref_texts,
                lang="en", 
                device=self.device,
                batch_size=4 # Ép chạy theo batch để tối ưu tốc độ
            )
            
            bert_f1 = np.mean(results['f1']) * 100
            bert_precision = np.mean(results['precision']) * 100
            bert_recall = np.mean(results['recall']) * 100
            
        except Exception as e:
            print(f"[Error] BERTScore computation failed: {e}")
            bert_f1 = bert_precision = bert_recall = 0.0

        # Chỉ trả về các chỉ số của BERTScore
        return {
            "BERTScore-Precision": bert_precision,
            "BERTScore-Recall": bert_recall,
            "BERTScore-F1": bert_f1
        }