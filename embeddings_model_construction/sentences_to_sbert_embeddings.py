"""
Generate 384-d or 768-d SBERT embeddings for your command sentences.

Input:
  - CSV from Step 1 (e.g. sbert_potentialfield_commands_2000_sentences_3_raters_decoupled_amp_radius.csv)
    This file has 6000 rows (3 raters × 2000 sentence_id), but only 2000 unique sentences.

Output:
  - embeddings_xxx.npz  (compressed)
      - sentence_id: (N,) array of ids (dtype=object)
      - embeddings: (N, 768) float32 matrix
  - embeddings_xxx.csv  (optional; large but convenient)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def main(input_csv: str, model_name: str, batch_size: int, normalize: bool, write_csv: bool):
    in_path = Path(input_csv)

    df = pd.read_csv(in_path)

    # Keep exactly one row per sentence_id (since embeddings depend only on the sentence text)
    unique = (
        df[["sentence_id", "sentence"]]
        .drop_duplicates(subset=["sentence_id"])
        .sort_values("sentence_id")
        .reset_index(drop=True)
    )

    sentence_ids = unique["sentence_id"].astype(str).tolist()
    sentences = unique["sentence"].astype(str).tolist()

    print(f"Loaded {len(df)} rows from CSV.")
    print(f"Found {len(sentences)} unique sentences (by sentence_id).")

    # Load the SBERT model
    model = SentenceTransformer(model_name)

    # Encode sentences -> (N, 768 or 384)
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,  # optional; often better OFF for regression
    ).astype(np.float32)

    # Sanity check: ensure correct dims
    if embeddings.ndim != 2 or not (embeddings.shape[1] == 768 or embeddings.shape[1] == 384):
        raise RuntimeError(
            f"Expected embeddings shape (N, 768 or 384), got {embeddings.shape}. "
            f"Model used: {model_name}"
        )

    # Save compressed NPZ (recommended)
    npz_file = "embeddings_" + str(embeddings.shape[1]) + ".npz"
    np.savez_compressed(
        npz_file,
        sentence_id=np.array(sentence_ids, dtype=object),
        embeddings=embeddings,
    )
    print(f"Saved: {npz_file}  (sentence_id + embeddings)")

    # Optional: write embeddings to CSV (bigger file; easy to open in pandas/excel)
    if write_csv:
        emb_cols = [f"emb_{i:03d}" for i in range(embeddings.shape[1])]
        out_df = pd.DataFrame(embeddings, columns=emb_cols)
        out_df.insert(0, "sentence_id", sentence_ids)
        out_df.insert(1, "sentence", sentences)
        csv_file = "embeddings_" + str(embeddings.shape[1]) + ".csv"
        out_df.to_csv(csv_file, index=False)
        print(f"Saved: {csv_file}")

    print("Done.")


if __name__ == "__main__":
    main(
        input_csv="sbert_potentialfield_commands_2000_sentences_3_raters_learnable.csv",
        #model_name="sentence-transformers/all-MiniLM-L6-v2", #384 dim
        model_name="sentence-transformers/all-mpnet-base-v2",  # 768-dim
        batch_size=64,
        normalize=False,
        write_csv=True,
    )
