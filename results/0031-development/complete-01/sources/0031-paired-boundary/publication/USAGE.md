# 0031 text export

This standalone standard-library publisher only consumes saved text. Run it once
after calibration, arithmetic qualification and fresh screen02 are completed with
clean bound outer receipts. Attempt01 must retain its failed status.

From the task root:

```sh
python3 -B work/0031-paired-boundary/publication/publish.py \
  --output work/0031-paired-boundary/publication-complete-01
```

The destination must not exist. Preserve command, source SHA, stdout, stderr and
exit status in a fresh publication-operation receipt. A failed export is not
silently replaced; review before using a fresh attempt path.

The export retains all 96 ranking files and 96 diagnostics, calibration reports,
both screen attempts, manufactured checks, sources, scope/admission/prior notes,
the saved closure decision and prior PR67 CI metadata. It never loads `.pt`,
`.npz`, images or models. JSON compaction preserves parsed values and keys;
literal path aliases are separately declared and original/public hashes recorded.
Existing payload references are not dereferenced. This is publication checking,
not an independent numerical audit or a new experiment. The repository is not
modified. Intended later public destination:
`results/0031-development/complete-01/`.
