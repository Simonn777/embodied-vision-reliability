"""Outcome-independent exact-duplicate grouping and deterministic allocation."""
import hashlib
import math
import numpy as np


def digest_array(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()


def allocate(entries, frame_hashes, seed=20261019):
    n = len(entries)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    edges = {}
    def link(i, j, reason):
        a, b = sorted((i, j))
        if a == b: return
        parent[find(b)] = find(a)
        edges.setdefault((a, b), set()).add(reason)
    for field in ["video_sha256", "prefix64_sha256"]:
        seen = {}
        for i, e in enumerate(entries):
            key = e[field]
            if key in seen: link(i, seen[key], field)
            else: seen[key] = i
    owner = {}
    for i, hashes in enumerate(frame_hashes):
        for h in set(hashes):
            if h in owner: link(i, owner[h], "shared_exact_rgb_frame")
            else: owner[h] = i
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)
    result = []
    for members in groups.values():
        group_id = "g-" + "-".join(f"{i:03}" for i in members)
        key = hashlib.sha256(f"0002|{seed}|{group_id}".encode()).hexdigest()
        result.append({"group_id": group_id, "members": members, "allocation_hash": key})
    result.sort(key=lambda g: (g["allocation_hash"], g["group_id"]))
    counts = {"development": 0, "calibration": 0, "evaluation": 0, "reserve": 0}
    for g in result:
        if 0 in g["members"]: split = "development"
        elif counts["calibration"] < 10: split = "calibration"
        elif counts["evaluation"] < 20: split = "evaluation"
        else: split = "reserve"
        g["partition"] = split
        counts[split] += len(g["members"])
    # Allocate dev group first without consuming either held-out target.
    by_index = {i: g for g in result for i in g["members"]}
    return {"groups": result, "counts": counts,
            "edges": [{"a": a, "b": b, "reasons": sorted(r)} for (a,b),r in sorted(edges.items())],
            "allocation": [{"archive_index": i, "group_id": by_index[i]["group_id"],
                            "partition": by_index[i]["partition"],
                            "development_inference_allowed": i == 0} for i in range(n)],
            "seed": seed, "scene_episode_independence": "unknown",
            "targets_met": counts["calibration"] >= 10 and counts["evaluation"] >= 20}
