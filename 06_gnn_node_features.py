# gnn_node_features.py
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import config


class NodeFeatureBuilder:
    def __init__(self):
        self.text_encoder = SentenceTransformer("jhgan/ko-sroberta-multitask")
        self.class_list = list(config.ALLOWED_CLASS_NAMES) + ["기타"]
        self.class_to_idx = {c: i for i, c in enumerate(self.class_list)}

    def encode_text(self, text):
        if text is None:
            return torch.zeros(768)

        # float NaN → "" 로 치환
        if isinstance(text, float):
            return torch.zeros(768)

        text = str(text).strip()
        if text == "" or text.lower() == "nan":
            return torch.zeros(768)

        return torch.tensor(self.text_encoder.encode(text))


    def onehot_class(self, cls):
        vec = torch.zeros(len(self.class_list))
        idx = self.class_to_idx.get(cls, self.class_to_idx["기타"])
        vec[idx] = 1
        return vec

    def layout_feature(self, row):
        # normalize
        return torch.tensor([
            row["x"]/3000, row["y"]/4000,
            row["w"]/3000, row["h"]/4000
        ])

    def build_node_feature(self, row):
        t = self.encode_text(row["text"])
        c = self.onehot_class(row["class"])
        l = self.layout_feature(row)

        v_flag = torch.tensor([1 if row["class"] in ["표", "차트"] else 0])

        return torch.cat([t, c, l, v_flag], dim=0)


def build_features_for_all():
    base = config.OUTPUT_DIR / "05_gnn_dataset" / "nodes"
    out = config.OUTPUT_DIR / "06_node_features"
    out.mkdir(parents=True, exist_ok=True)

    builder = NodeFeatureBuilder()

    for csv in base.glob("*_nodes.csv"):
        df = pd.read_csv(csv)
        feats = []

        for _, row in df.iterrows():
            f = builder.build_node_feature(row)
            feats.append(f.unsqueeze(0))

        tensor = torch.cat(feats, dim=0)
        torch.save(tensor, out / f"{csv.stem}.pt")
        print(f"[OK] {csv.stem} → node feature 저장 완료")


if __name__ == "__main__":
    build_features_for_all()
