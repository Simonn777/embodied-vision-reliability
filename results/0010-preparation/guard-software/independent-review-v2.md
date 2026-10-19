# Cached guard runtime delta: independent approval

No blocking finding in the sole runtime delta from baseline-v1. Reviewed `guard.py` SHA256 `546c89da76a61640d473867499566f5b17e793058956cba6f0ba9fd6ca0195ce` and tests SHA256 `1158bac06aade9a3cd2d4db46df827326bdd541e0cca117523a45d4805f74819`. Preservation pins remain unchanged (`20a8aabd3f501d2475610ef8d8fd3b51e135acd74eaa2226426a1a2cbc906258`).

The added function runs only after the 65 pinned evidence files pass their existing hash checks. It reads the actual retained imports schema (133 overlay entries with bytes/SHA256; nine runtime entry hashes), rejects changed interpreter/PYTHONPATH, hashes the recorded files without importing them, checks the original import working directory, and includes these file identities in the /proc holder and final stat-stability checks. Artificial tampering of either an overlay or module entry is rejected. No acquisition, installation, model import or forward is added. This closes the earlier recorded-runtime identity boundary for those stated files.

The scope remains explicit: 133 named overlay files and nine module entries, not all transitive system libraries or an extra-file search. Original freeze/CI verification and launching the unchanged probe with this same interpreter/PYTHONPATH remain the caller's responsibility. A snapshot cannot guarantee future immutability. No broad new runtime guarantee is claimed.

Independent `python3 -m unittest -v test_guard` rerun: actual exit 0; 11 artificial tests passed, no skips. The actual stderr log is `/tmp/evr-0010-cached-guard-v2-review.stderr.log`; stdout is empty. All tests use fake filesystem/process fixtures. No real cloud/assets/model were accessed, and no repository or author source was edited.

Stderr: 900 bytes; SHA256 `a22df204abccf467d5b1d59290fefd338ef6721ee6adf0b3bc75b66370c61773`.
