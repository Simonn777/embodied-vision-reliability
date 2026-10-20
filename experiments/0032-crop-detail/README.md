# Fixed current-frame crop baseline

This is a simplified known-method adaptation, closed after the full development evaluation lost3.7811 J&F points. See [complete results](../../results/0032-development/complete-01/README.md), [prospective scope](SCOPE.md) and [metric scope](METRIC_SCOPE.md).

The six manufactured geometry tests can run without SAM2 or datasets:

```sh
python -B -W error experiments/0032-crop-detail/test_refine.py -v
```

They require NumPy and OpenCV. Scientific workers require the declared qualified CPU runtime, frozen pinned SAM2.1 tiny checkpoint, exact existing cache/index and authorized annotation inputs. Restore the original work/0032-crop-detail and work/0030-memory-content layout from source-bound files; public ${WORKSPACE} paths are aliases. A bare repository clone does not contain the private model/data/cache. Do not replace inputs, omit frames, or treat tests as model efficacy evidence.
