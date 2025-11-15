# clean_edges.py
import re
import pandas as pd
from pathlib import Path
import config

#########################################################
# 1) tensor(xx) → int 변환 함수
#########################################################
def parse_tensor_index(val):
    """
    src,dst가 'tensor(7, device=cuda)' 형태일 수 있으므로 int로 변환한다.
    """
    if isinstance(val, int):
        return val

    if isinstance(val, str):
        # 정수만 추출
        match = re.search(r"\d+", val)
        if match:
            return int(match.group(0))

    # 변환 실패 시 -1
    return -1


#########################################################
# 2) edge CSV 클린 함수
#########################################################
def clean_edge_csv(input_csv: Path, output_csv: Path,
                   top_k=3,
                   semantic_threshold=0.25,
                   dxdy_threshold=0.001):

    df = pd.read_csv(input_csv)

    # tensor 문자열 → int
    df["src"] = df["src"].apply(parse_tensor_index)
    df["dst"] = df["dst"].apply(parse_tensor_index)

    # self-loop 제거
    df = df[df["src"] != df["dst"]]

    # semantic_sim 기반 필터 (visual_link=1은 무조건 보존)
    df = df[(df["semantic_sim"] >= semantic_threshold) | (df["visual_link"] == 1)]

    # spatial noise 제거
    df = df[(df["dx"].abs() > dxdy_threshold) | (df["dy"].abs() > dxdy_threshold)]

    if len(df) == 0:
        print(f"[!] After filtering → No edges left → DELETE: {input_csv.name}")
        return None  # 삭제 플래그

    # ----------------------
    # top-k per src 적용
    # ----------------------
    cleaned_rows = []
    for src, group in df.groupby("src"):
        group = group.sort_values(
            by=["visual_link", "semantic_sim"],
            ascending=[False, False]
        )
        k = min(top_k, len(group))
        cleaned_rows.append(group.head(k))

    if len(cleaned_rows) == 0:
        print(f"[!] All groups empty → DELETE: {input_csv.name}")
        return None

    df_clean = pd.concat(cleaned_rows, ignore_index=True)
    df_clean = df_clean.sort_values(by=["src", "dst"]).reset_index(drop=True)

    # 저장
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✔ Cleaned saved → {output_csv.name}")
    return True


#########################################################
# 3) 메인 실행 함수
#########################################################
def main():
    edges_root = config.OUTPUT_DIR / "05_gnn_dataset" / "edges"
    output_root = config.OUTPUT_DIR / "06_clean_edges"
    output_root.mkdir(parents=True, exist_ok=True)

    all_files = sorted(edges_root.glob("*.csv"))
    print(f"총 {len(all_files)}개 edge 파일 정리 시작.")

    kept = 0
    removed = 0

    for f in all_files:
        out_f = output_root / f.name
        ok = clean_edge_csv(f, out_f)

        if ok is None:
            removed += 1
            continue
        kept += 1

    print("=" * 40)
    print(f"정리 완료: 유지 {kept}개, 삭제 {removed}개")
    print("=" * 40)


if __name__ == "__main__":
    main()
