"""Buoc 2: gop cac file Tatoeba (moi ngon ngu mot file) thanh mot file CSV can bang."""
import glob
import sys

import pandas as pd

MAX_PER_LANG = 5000  # lay toi da bay nhieu cau moi ngon ngu de du lieu can bang

files = sorted(glob.glob("data/*.tsv*"))
if not files:
    sys.exit("Khong tim thay file nao trong thu muc data/ (can file .tsv hoac .tsv.bz2)")

frames = []
for path in files:
    # quoting=3 (QUOTE_NONE): tranh loi khi cau co dau nhay kep
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["id", "lang", "text"],
        quoting=3,
        dtype=str,
    )
    df = df.dropna(subset=["lang", "text"])
    if len(df) > MAX_PER_LANG:
        df = df.sample(MAX_PER_LANG, random_state=42)
    print(f"{path}: {len(df)} cau, ngon ngu = {df['lang'].unique().tolist()}")
    frames.append(df[["lang", "text"]])

data = pd.concat(frames, ignore_index=True)
data = data.drop_duplicates(subset="text")
data.to_csv("data/sentences.csv", index=False)

print("\nTong so cau:", len(data))
print(data["lang"].value_counts())