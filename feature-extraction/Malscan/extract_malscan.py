#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified pipeline: APK -> (in-memory) call graph -> MaMaDroid + MalScan features
- Skips APKs already processed (resume-safe)
- Parallel processing with multiprocessing
- No graph files written to disk
- Aggregates to CSVs at the end

Dependencies:
  pip install androguard networkx numpy

Notes:
  - Ensure your sensitive_apis.txt uses the same node naming as produced here:
    "Landroid/telephony/SmsManager;->sendTextMessage(...)" (smali-style with '->')
  - family_list.txt should list API families (e.g., android., java., com., org., javax.)
"""

import os
import sys
import csv
import gc
import time
import glob
import json
import math
import hashlib
import zipfile
import logging
import argparse
import traceback
from functools import partial
from multiprocessing import get_start_method
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx
from androguard.misc import AnalyzeAPK


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pipeline")


# ----------------------------
# Helpers
# ----------------------------

def sha256_of_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def list_apks(input_path):
    """
    Returns: list of (apk_path, label) tuples
    label: 0 for benign, 1 for malware, or None if unlabeled
    """
    if os.path.isdir(input_path):
        # Detect dataset structure
        benign_dir = os.path.join(input_path, "benign")
        malware_dir = os.path.join(input_path, "malware")
        if os.path.isdir(benign_dir) and os.path.isdir(malware_dir):
            apks_b = glob.glob(os.path.join(benign_dir, "*.apk"))
            apks_m = glob.glob(os.path.join(malware_dir, "*.apk"))
            return [(p, 0) for p in apks_b] + [(p, 1) for p in apks_m]
        else:
            # Flat directory
            apks = glob.glob(os.path.join(input_path, "*.apk"))
            return [(p, None) for p in apks]
    else:
        # Single file
        if input_path.endswith(".apk"):
            return [(input_path, None)]
        else:
            raise ValueError(f"Not an APK: {input_path}")


def load_sensitive_apis(path):
    sensitive_apis = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                sensitive_apis.append(s)
    return sensitive_apis


def load_families(path):
    families = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fam = line.strip()
            if fam:
                families.append(fam)
    # Add two abstractions as per MaMaDroid
    families.append("self-defined")
    families.append("obfuscated")
    return families


def smali_to_java_class_name(smali_class_name):
    """
    Convert 'Landroid/telephony/SmsManager;' -> 'android.telephony.SmsManager'
    """
    if not smali_class_name.startswith("L") or not smali_class_name.endswith(";"):
        # best effort fallback
        return smali_class_name.replace("/", ".").lstrip("L").rstrip(";")
    return smali_class_name[1:-1].replace("/", ".")


def call_graph_from_dx(dx):
    """
    Build a DiGraph using node names consistent with your prior code:
    'Lpkg/Class;->method(desc)ret'
    """
    CG = nx.DiGraph()
    nodes = dx.find_methods('.*', '.*', '.*', '.*')
    for m in nodes:
        API = m.get_method()
        class_name = API.get_class_name()  # e.g., Landroid/...
        method_name = API.get_name()
        descriptor = API.get_descriptor()

        api_call = f"{class_name}->{method_name}{descriptor}"

        # Skip methods without outgoing xrefs
        xrefs = m.get_xref_to()
        if not xrefs:
            continue

        CG.add_node(api_call)

        for _, callee, _ in xrefs:
            c_cls = callee.get_class_name()
            c_mname = callee.get_method().get_name()
            c_desc = callee.get_descriptor()
            _callee = f"{c_cls}->{c_mname}{c_desc}"

            CG.add_node(_callee)
            if not CG.has_edge(api_call, _callee):
                CG.add_edge(api_call, _callee)

    return CG


def is_obfuscated_or_self_defined(java_class_name):
    items = java_class_name.split(".")
    # Heuristic consistent with your original code:
    short_count = sum(1 for it in items if len(it) < 3)
    if short_count > (len(items) / 2):
        return "obfuscated"
    return "self-defined"


def smali_node_to_family(node_name, families):
    """
    node_name is like: 'Landroid/..;->method(desc)'
    We abstract by family on the *class* (caller/callee class)
    """
    smali_class = node_name.split("->")[0]  # Landroid/..
    java_class = smali_to_java_class_name(smali_class)  # android....

    for fam in families[:-2]:  # exclude the last two appended labels
        if java_class.startswith(fam):
            return fam

    # Fall back to heuristic
    return is_obfuscated_or_self_defined(java_class)



def get_centrality_vector(CG, sensitive_apis, kind="degree", katz_alpha=0.005, katz_max_iter=2000, katz_tol=1e-6):
    """
    Compute centrality for each node and return a vector aligned to sensitive_apis.
    Directed graph centralities as per NetworkX functions.
    """
    # Edge case: empty graph
    if CG is None or CG.number_of_nodes() == 0:
        return np.zeros(len(sensitive_apis), dtype=np.float64)

    try:
        if kind == "degree":
            cent = nx.degree_centrality(CG)
        elif kind == "katz":
            # For directed graphs with potential convergence issues, keep small alpha.
            cent = nx.katz_centrality(CG, alpha=katz_alpha, max_iter=katz_max_iter, tol=katz_tol)
        elif kind == "closeness":
            cent = nx.closeness_centrality(CG)  # directed closeness by default
        elif kind == "harmonic":
            cent = nx.harmonic_centrality(CG)
        else:
            raise ValueError(f"Unknown centrality: {kind}")
    except Exception as e:
        logger.warning(f"Centrality computation failed for {kind}: {e}")
        return np.zeros(len(sensitive_apis), dtype=np.float64)

    vec = np.fromiter((cent.get(api, 0.0) for api in sensitive_apis), dtype=np.float64, count=len(sensitive_apis))
    return vec


# ----------------------------
# Worker (per APK)
# ----------------------------
from scipy.sparse import csr_matrix, save_npz
def process_apk_worker(
    apk_path,
    label,
    out_root,
    family_list,
    sensitive_apis,
    centralities,
    katz_alpha,
    katz_max_iter,
    katz_tol,
):
    """
    Returns summary dict:
      {
        "sha256": ..., "apk": ..., "label": 0/1/None, 
        "malscan_done": {"degree": True/False, ...}
      }
    """
    summary = {
        "sha256": None,
        "apk": apk_path,
        "label": label,
        "malscan_done": {c: False for c in centralities},
        "error": None,
    }
    try:
        # Quick checks
        if not zipfile.is_zipfile(apk_path) or not apk_path.endswith(".apk"):
            summary["error"] = "Not an APK/zip file"
            return summary

        file_sha = sha256_of_file(apk_path)
        summary["sha256"] = file_sha

        # Paths for outputs
        ms_root = os.path.join(out_root, "features", "malscan")
        ms_files = {}
        for c in centralities:
            cdir = os.path.join(ms_root, c)
            os.makedirs(cdir, exist_ok=True)
            ms_files[c] = os.path.join(cdir, f"{file_sha}.npy")

        # Resume check: if everything exists, skip heavy work
        need_any_malscan = any(not os.path.isfile(p) for p in ms_files.values())
        if not need_any_malscan:
            for c in centralities:
                summary["malscan_done"][c] = True
            return summary

        # Heavy step: analyze and build call graph once
        a, d, dx = AnalyzeAPK(apk_path)
        CG = call_graph_from_dx(dx)

        # MalScan features (per centrality)
        for c in centralities:
            if os.path.isfile(ms_files[c]):
                summary["malscan_done"][c] = True
                continue
            vec = get_centrality_vector(
                CG, sensitive_apis, kind=c,
                katz_alpha=katz_alpha, katz_max_iter=katz_max_iter, katz_tol=katz_tol
            )
            #np.save(ms_files[c], vec)
            vec = vec.astype(np.float32)
            vec_sparse = csr_matrix(vec)     # <-- 变成稀疏
            save_npz(ms_files[c], vec_sparse) # <-- 稀疏保存
            summary["malscan_done"][c] = True

        # Free memory
        del CG, a, d, dx
        gc.collect()

    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        logger.error(f"[FAIL] {apk_path}: {e}")
        logger.debug(traceback.format_exc())

    return summary

# ----------------------------
# Main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Unified pipeline: in-memory graph -> MalScan features")
    p.add_argument("-i", "--input", required=True, type=str,
                   help="Path to an APK file OR a directory. If directory contains benign/ and malware/ subdirs, labels are auto-detected.")
    p.add_argument("-o", "--output", required=True, type=str, help="Output directory (will be created if missing).")
    p.add_argument("--centralities", default="degree,katz,closeness,harmonic",
                   help="Comma-separated centralities to compute. Options: degree,katz,closeness,harmonic")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1), help="Number of worker processes.")
    p.add_argument("--katz-alpha", type=float, default=0.005, help="Katz alpha (smaller => easier convergence).")
    p.add_argument("--katz-max-iter", type=int, default=2000, help="Katz max iterations.")
    p.add_argument("--katz-tol", type=float, default=1e-6, help="Katz tolerance.")
    return p.parse_args()


def main():
    # Ensure proper start method on some platforms
    try:
        if get_start_method(allow_none=True) != "spawn":
            import multiprocessing as mp
            mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    centralities = [c.strip().lower() for c in args.centralities.split(",") if c.strip()]
    for c in centralities:
        if c not in {"degree", "katz", "closeness", "harmonic"}:
            raise ValueError(f"Unsupported centrality: {c}")

    logger.info("Loading configuration files...")
    sensitive_apis = load_sensitive_apis("sensitive_apis.txt")
    family_list = load_families("families.txt")

    apk_entries = list_apks(args.input)
    if not apk_entries:
        logger.error("No APKs found.")
        sys.exit(1)

    logger.info(f"Found {len(apk_entries)} APK(s). Starting parallel extraction with {args.workers} workers...")

    records = []  # for aggregation
    start = time.time()

    worker = partial(
        process_apk_worker,
        out_root=args.output,
        family_list=family_list,
        sensitive_apis=sensitive_apis,
        centralities=centralities,
        katz_alpha=args.katz_alpha,
        katz_max_iter=args.katz_max_iter,
        katz_tol=args.katz_tol,
    )

    successes = 0
    skipped = 0
    failures = 0

    # Submit jobs
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(worker, apk_path, lbl): (apk_path, lbl)
            for (apk_path, lbl) in apk_entries
        }
        for fut in as_completed(futures):
            res = fut.result()
            # Progress accounting
            if res.get("error"):
                failures += 1
                logger.error(f"[{failures} fails] {res['apk']}: {res['error']}")
            else:
                all_done = all(res["malscan_done"].values())
                if all_done:
                    successes += 1
                else:
                    # allowed: some parts previously existed (resume)
                    skipped += 1
                logger.info(f"[OK] {res['apk']} -> sha256={res['sha256']} malscan={res['malscan_done']}")
            # Collect record for CSV aggregation
            records.append({"sha256": res.get("sha256"), "label": res.get("label")})


    dur = time.time() - start
    logger.info(f"Done in {dur:.1f}s | Success: {successes} | Partial/Skipped: {skipped} | Failures: {failures}")


if __name__ == "__main__":
    main()
