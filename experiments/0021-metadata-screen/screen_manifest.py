"""Metadata-only candidate image-token graph. Never opens HDF5 or scientific values."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


PATTERN = re.compile(r"^([0-9]+_[0-9]+_o)_([0-9]+_[0-9]+_o)$")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(key):
    match = PATTERN.fullmatch(key)
    if match is None:
        raise ValueError(f"Unrecognized candidate token syntax: {key!r}")
    a, b = match.groups()
    if a == b:
        raise ValueError(f"Self-pair not supported: {key!r}")
    return a, b


def graph(keys):
    adjacency = {}
    edges = set()
    for key in keys:
        a, b = tokens(key)
        edges.add(tuple(sorted((a, b))))
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return adjacency, edges


def query_support(keys, reference_keys):
    adjacency, _ = graph(reference_keys)
    rows = []
    for key in keys:
        a, b = tokens(key)
        left = adjacency.get(a, set()) - {b}
        right = adjacency.get(b, set()) - {a}
        rows.append({
            "key": key, "candidate_tokens": [a, b],
            "reference_neighbor_counts": [len(left), len(right)],
            "both_endpoints_have_reference_edge": bool(left and right),
            "common_reference_neighbor_count": len(left & right),
            "four_distinct_vertices_possible": any(c != d for c in left for d in right),
        })
    n = len(rows)
    counts = {
        "query_count": n,
        "both_endpoints_have_reference_edge_count": sum(r["both_endpoints_have_reference_edge"] for r in rows),
        "common_reference_neighbor_query_count": sum(r["common_reference_neighbor_count"] > 0 for r in rows),
        "four_distinct_vertices_query_count": sum(r["four_distinct_vertices_possible"] for r in rows),
    }
    for key in tuple(counts):
        if key != "query_count":
            counts[key.replace("_count", "_fraction")] = counts[key] / n if n else None
    return {"summary": counts, "queries": rows}


def topology(keys):
    adjacency, edges = graph(keys)
    degrees = [len(v) for v in adjacency.values()]
    unseen = set(adjacency)
    sizes = []
    while unseen:
        stack = [min(unseen)]
        unseen.remove(stack[0])
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            fresh = adjacency[node] & unseen
            unseen.difference_update(fresh)
            stack.extend(sorted(fresh))
        sizes.append(size)
    triangles = sum(len(adjacency[a] & adjacency[b]) for a, b in edges) // 3
    assert sum(degrees) == 2 * len(edges)
    return {
        "raw_pair_key_count": len(keys), "unique_undirected_edge_count": len(edges),
        "duplicate_undirected_edge_count": len(keys) - len(edges),
        "scene_qualified_candidate_node_count": len(adjacency),
        "degree_min": min(degrees) if degrees else None,
        "degree_max": max(degrees) if degrees else None,
        "degree_mean": sum(degrees) / len(degrees) if degrees else None,
        "degree_histogram": dict(sorted(Counter(degrees).items())),
        "node_degree": {node: len(adjacency[node]) for node in sorted(adjacency)},
        "component_sizes_descending": sorted(sizes, reverse=True),
        "triangle_count": triangles,
    }


def screen(manifest):
    if manifest["schema"] != "evr0017.development.v1":
        raise ValueError("Unexpected manifest schema")
    selected = manifest["selected_pairs"]
    if len({(r["scene"], r["key"]) for r in selected}) != len(selected):
        raise ValueError("Duplicate selected pair")
    scenes = []
    for scene in manifest["scenes"]:
        name = scene["scene"]
        keys = scene["root_keys"]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate root key")
        query = [r["key"] for r in selected if r["scene"] == name]
        if not set(query).issubset(keys):
            raise ValueError("Selected key absent from metadata")
        references = sorted(set(keys) - set(query))
        shapes = Counter()
        for pair in scene["pairs_metadata"]:
            datasets = pair["datasets"]
            shapes[tuple(datasets["depth1"]["shape"])] += 1
            shapes[tuple(datasets["depth2"]["shape"])] += 1
            for side in (1, 2):
                if datasets[f"depth{side}"]["shape"] != [datasets[f"x{side}"]["shape"][0]]:
                    raise ValueError("Depth/match metadata shape disagreement")
        scenes.append({
            "scene": name, "filename": scene["filename"],
            "all_root_keys_topology": topology(keys),
            "selected_keys_only_topology": topology(query),
            "all_edges_as_queries_other_edges_as_references": query_support(keys, keys)["summary"],
            "selected_queries_all_other_edges_available": query_support(query, keys),
            "selected_queries_all_selected_edges_excluded_from_reference_pool": query_support(query, references),
            "selected_queries_selected_edges_only_references": query_support(query, query)["summary"],
            "nonselected_reference_key_count": len(references),
            "depth_vector_shape_histogram_both_sides": {str(shape): n for shape, n in sorted(shapes.items())},
        })
    aggregate = {}
    for mode in ("all_edges_as_queries_other_edges_as_references", "selected_queries_all_other_edges_available", "selected_queries_all_selected_edges_excluded_from_reference_pool", "selected_queries_selected_edges_only_references"):
        summaries = [s[mode] if "summary" not in s[mode] else s[mode]["summary"] for s in scenes]
        totals = {key: sum(row[key] for row in summaries) for key in summaries[0] if key.endswith("_count")}
        for key in tuple(totals):
            if key != "query_count":
                totals[key.replace("_count", "_fraction")] = totals[key] / totals["query_count"]
        aggregate[mode] = totals
    aggregate["all_root_keys_topology"] = {
        field: sum(s["all_root_keys_topology"][field] for s in scenes)
        for field in ("raw_pair_key_count", "unique_undirected_edge_count", "scene_qualified_candidate_node_count", "triangle_count")
    }
    return {
        "schema": "evr0021.metadata-graph.v1",
        "input_access": {"manifest_only": True, "hdf5_opened": False, "dataset_values_read": False, "images_read": False, "ground_truth_read": False},
        "identity_status": "unverified_candidate_tokens_not_canonical_image_ids",
        "token_parser": PATTERN.pattern,
        "scope": "Scene-qualified undirected simple token graph; shared strings across scenes are not merged.",
        "reference_definition": "Query edge excluded. Reference-edge availability is combinatorial, not statistical independence, parallax, common pixel support, or validated shared depth.",
        "depth_schema_limitation": "Manifest shapes describe per-correspondence depth vectors, not per-image dense maps or provenance hashes.",
        "allocation": "None. Existing selected pairs are only descriptive query cohorts; no new scientific allocation.",
        "aggregate": aggregate, "scenes": scenes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.name != "0017-development-manifest.json":
        raise ValueError("This metadata-only script accepts only the authorized manifest filename")
    if args.output.exists():
        raise FileExistsError(args.output)
    record = screen(json.loads(args.manifest.read_text()))
    record["manifest_sha256"] = digest(args.manifest)
    record["script_sha256"] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(record, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(record["aggregate"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
