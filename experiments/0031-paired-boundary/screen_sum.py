"""Technical retry: unchanged pairing screen, explicit bounded cosine reductions."""
import json
from pathlib import Path
import sys
import screen


def scores(xy_descriptors, bank, permutation):
    import numpy as np
    x, y = xy_descriptors
    u, v = bank[:, 0], bank[:, 1]
    # Same float64 dot products, without the strided BLAS matmul path.
    xu = np.sum(u*x[None, :], axis=1, dtype=np.float64)
    xv = np.sum(v*x[None, :], axis=1, dtype=np.float64)
    yu = np.sum(u*y[None, :], axis=1, dtype=np.float64)
    yv = np.sum(v*y[None, :], axis=1, dtype=np.float64)
    screen.require(all(np.isfinite(a).all() for a in (xu, xv, yu, yv)), "nonfinite cosine reductions")
    plus, minus = xu+yv, xv+yu
    pp, pm = float(np.max(plus)), float(np.max(minus))
    up, um = float(np.max(xu)+np.max(yv)), float(np.max(xv)+np.max(yu))
    dp, dm = xu+yv[permutation], xv[permutation]+yu
    return dict(paired=pp-pm, unpaired_max=up-um,
        deranged=float(np.max(dp)-np.max(dm))), dict(
        forward=pp, reverse=pm, forward_gap=up-pp, reverse_gap=um-pm,
        forward_index=int(np.argmax(plus)), reverse_index=int(np.argmax(minus)))


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text())
    screen.require(spec["source_sha256"].get(str(Path(__file__).resolve())) ==
                   screen.sha(__file__), "technical adapter source binding")
    screen.scores = scores
    screen.main(sys.argv[1])
