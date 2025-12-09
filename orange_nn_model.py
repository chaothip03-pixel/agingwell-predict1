# orange_nn_model.py
# ใช้ไฟล์ nn_export.json (sklearn-like export) เพื่อทำ inference โดยไม่ต้องใช้ Orange/ sklearn runtime
import json
import numpy as np
from typing import List, Union

class OrangeNNModel:
    def __init__(self, json_path: str):
        """
        โหลดน้ำหนักจาก nn_export.json
        คาดว่าไฟล์มี keys: 'coefs' (list of 2D lists), 'intercepts' (list of 1D lists) และ 'classes' (list)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # รับ coefs & intercepts แบบยืดหยุ่น
        self.coefs = [np.array(layer) for layer in data.get('coefs', [])]
        self.intercepts = [np.array(b) for b in data.get('intercepts', [])] if 'intercepts' in data else [np.zeros(w.shape[1]) for w in self.coefs]
        self.classes_ = data.get('classes', None)  # e.g. [0.0, 1.0, 2.0]
        if self.classes_ is None:
            # fallback: numeric classes 0..(n_outputs-1)
            n_out = data.get('n_outputs', None)
            if n_out is not None:
                self.classes_ = list(range(n_out))
            else:
                # fallback final weight shape
                if len(self.coefs) > 0:
                    self.classes_ = list(range(self.coefs[-1].shape[1]))
                else:
                    self.classes_ = []

        self.n_features_in = data.get('n_features_in', None)
        self.activation = data.get('activation', 'relu')
        self.out_activation = data.get('out_activation', 'softmax')
        # default mapping of features (index order). Edit if you have explicit feature names.
        self.feature_order = data.get('feature_order', None)  # optional: list of names
        # If feature_order not provided, assume indices 0..n_features_in-1

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        # stable softmax along last axis
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def _forward_single(self, x: np.ndarray) -> np.ndarray:
        """
        x: shape (n_features,)
        returns: logits/prob vector (n_classes,)
        """
        a = x
        for i, W in enumerate(self.coefs):
            b = self.intercepts[i] if i < len(self.intercepts) else np.zeros(W.shape[1])
            # note: sklearn MLP uses X @ W + b (W shape: n_in x n_out)
            a = a @ W + b
            # if not last layer apply activation
            if i < len(self.coefs) - 1:
                if self.activation == 'relu':
                    a = self._relu(a)
                else:
                    # only relu implemented for now
                    a = self._relu(a)
        # final activation
        if self.out_activation == 'softmax':
            probs = self._softmax(a.reshape(1, -1))[0]
        else:
            # fallback: softmax
            probs = self._softmax(a.reshape(1, -1))[0]
        return probs

    def predict_proba(self, X: Union[List[List[float]], List[float], np.ndarray]) -> np.ndarray:
        """
        X can be:
         - single sample: list of length n_features
         - batch: list of samples or numpy array shape (n_samples, n_features)
        returns: np.ndarray shape (n_samples, n_classes)
        """
        arr = np.array(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        probs = np.vstack([self._forward_single(row) for row in arr])
        return probs

    def predict(self, X: Union[List[List[float]], List[float], np.ndarray]) -> List:
        probs = self.predict_proba(X)
        idxs = np.argmax(probs, axis=1)
        return [self.classes_[i] for i in idxs]

    def predict_with_confidence(self, X):
        """
        returns list of dicts: {'class': class_label, 'confidence': float}
        """
        probs = self.predict_proba(X)
        out = []
        for p in probs:
            i = int(np.argmax(p))
            out.append({'class': self.classes_[i], 'confidence': float(p[i]), 'proba': p.tolist()})
        return out

# Example usage (when run directly)
if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'nn_export.json'
    m = OrangeNNModel(model_path)
    # demo single sample of zeros (modify to real features)
    sample = np.zeros(m.coefs[0].shape[0]) if len(m.coefs) > 0 else np.zeros(m.n_features_in or 0)
    print("Predict demo:", m.predict_with_confidence(sample))
