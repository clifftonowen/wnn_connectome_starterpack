"""Stateful ONNX model for the competition scorer. Copied into the ZIP as solution.py."""
from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort


class PredictionModel:
    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.use_per_session_threads = True
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(Path(__file__).resolve().parent / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        states = [item for item in self.session.get_inputs() if item.name != "x"]
        self.names = [item.name for item in states]
        self.zero = [np.zeros(item.shape, dtype=np.float32) for item in states]
        self.state = self.zero
        self.seq_ix = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.state = self.zero
        feed = dict(zip(self.names, self.state))
        feed["x"] = np.asarray(data_point.state, dtype=np.float32).reshape(1, 1, 112)
        out = self.session.run(None, feed)
        self.state = out[1:]
        if not data_point.need_prediction:
            return None
        return np.asarray(out[0][0, 0], dtype=np.float32)
