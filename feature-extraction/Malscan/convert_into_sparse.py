import os
import numpy as np
from scipy.sparse import csr_matrix, vstack, save_npz
from tqdm import tqdm

for i in ["closeness","degree","harmonic","katz"]:
    mam_dir = f"graphs/features/malscan/{i}/"
    out = f"graphs/malscan_{i}_sparse_added.npz"

    rows = []
    shas = []
    labels = []   # if you stored labels elsewhere, load here

    for f in tqdm(os.listdir(mam_dir)):
        if not f.endswith(".npy"):
            continue
        sha = f.replace(".npy", "")
        data = np.load(os.path.join(mam_dir, f))
        #vec = data["family_feature"]#this is for mamadroid
        rows.append(csr_matrix(data))
        shas.append(sha)
    #print(rows)
    big = vstack(rows)
    save_npz(out, big)
    np.save(f"graphs/malscan_{i}_shas_added.npy", np.array(shas))
