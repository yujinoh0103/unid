import pandas as pd
import re
import config
from pathlib import Path

def parse_tensor_index(x):
    """
    "tensor(7, device='cuda:0')" → 7
    int면 그대로 반환
    """
    if isinstance(x, int):
        return x

    if isinstance(x, str):
        m = re.search(r"tensor\((\d+)", x)
        if m:
            return int(m.group(1))
        # 만약 그냥 숫자 형태라면
        if x.isdigit():
            return int(x)

    # 실패 시 0 리턴 (혹은 에러로 처리 가능)
    return 0


def clean_edge_csv(input_csv, output_csv, top_k=3,
                   semantic_threshold=0.25,
                   dxdy_threshold=0.001):

    df = pd.read_csv(input_csv)

    # tensor → int 변환
    df["src"] = df["src"].apply(parse_tensor_index)
    df["dst"] = df["dst"].apply(parse_tensor_index)

    # self-loop 제거
    df = df[df["src"] != df["dst"]]

    # semantic 낮은 edge 제거 (visual_link=1은 무조건 살림)
    df = df[(df["semantic_sim"] >= semantic_threshold) | (df["visual_link"] == 1)]

    # spatial noise 제거
    df = df[(df["dx"].abs() > dxdy_threshold) | (df["dy"].abs() > dxdy_threshold)]

    # ---------- 핵심: src 별로 필터링 ----------
    cleaned = []
    for src, group in df.groupby("src"):
        if len(group) == 0:
            continue

        group = group.sort_values(
            by=["visual_link", "semantic_sim"],
            ascending=[False, False]
        )

        # top-k 적용
        k = min(top_k, len(group))
        cleaned.append(group.head(k))

    # --------------------------------------------
    # 빈 리스트일 경우: 빈 DF를 저장
    # --------------------------------------------
    if len(cleaned) == 0:
        print(f"[!] No edges left after cleaning → saving empty CSV: {output_csv}")

        empty_df = pd.DataFrame(columns=df.columns)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return

    # concat
    df_clean = pd.concat(cleaned, ignore_index=True)
    df_clean = df_clean.sort_values(by=["src", "dst"]).reset_index(drop=True)

    # 저장
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✔ Clean saved → {output_csv}")

def main():
    raw_dir = config.OUTPUT_DIR / "05_gnn_dataset" / "edges"
    clean_dir = config.OUTPUT_DIR / "05_gnn_dataset" / "edges_clean"

    print(f"🔍 Raw edge folder:   {raw_dir}")
    print(f"💾 Clean edge folder: {clean_dir}")

    raw_files = list(raw_dir.glob("*.csv"))

    if not raw_files:
        print("❌ No raw edge CSV found.")
        return

    for f in raw_files:
        output_f = clean_dir / f.name
        clean_edge_csv(f, output_f)

    print("🎉 All edges cleaned!")

if __name__ == "__main__":
    main()
