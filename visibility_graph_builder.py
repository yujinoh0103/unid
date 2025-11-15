# visibility_graph_builder.py
import pandas as pd
import numpy as np
from pathlib import Path

def build_visibility_graph(df_regions, long_edge_thresh=0.25):
    """
    df_regions에는 각 영역의 bbox 및 텍스트 히스토그램 등이 담겨 있어야 함.
    bbox 컬럼: x1, y1, x2, y2
    """
    boxes = df_regions[["x1","y1","x2","y2"]].values
    nodes = list(range(len(boxes)))
    edges = []

    for i, b1 in enumerate(boxes):
        for j, b2 in enumerate(boxes):
            if i == j:
                continue
            # 수직/수평 가시성 판단
            if _is_horiz_visible(b1, b2, boxes) or _is_vert_visible(b1, b2, boxes):
                # 긴 엣지 제거
                height = abs(b1[1] - b2[3])
                page_height = df_regions["page_height"].iloc[0]
                if height / page_height <= long_edge_thresh:
                    edges.append((i,j))
    return nodes, edges

def _is_horiz_visible(...):
    # 구현 생략, 위 논문 방식 참고
    pass

def _is_vert_visible(...):
    # 구현 생략
    pass
