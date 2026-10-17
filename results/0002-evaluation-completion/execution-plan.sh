set -e
for group in g-031 g-015 g-001 g-005 g-038 g-004 g-021 g-010 g-014 g-007 g-011 g-049 g-028 g-025 g-019 g-006 g-037 g-008 g-027 g-024
do
  short=${group/-/}
  if [ -f "results/0002-pilot-execution/${group}-20261017-postrun/publication_audit.json" ]; then
    python - "$group" <<'CHECK'
import json,sys
from pathlib import Path
r=json.loads((Path('results/0002-pilot-execution')/(sys.argv[1]+'-20261017-postrun/publication_audit.json')).read_text())
assert r['status']=='completed' and r['child_exit_code_observed_by_parent']==0
CHECK
    continue
  fi
  python scratch/prepare_evaluation.py "$group"
  python "scratch/supervise_${short}.py"
  audit="results/0002-pilot-execution/${group}-20261017"
  post="results/0002-pilot-execution/${group}-20261017-postrun"
  cp "scratch/verify_${short}.py" "$audit/verification-source.py"
  if [ "$group" = g-031 ]; then
    cp scratch/software_g031.py "$audit/software-checks-source.py"
    python scratch/postrun_g031.py software
  fi
  python "scratch/postrun_${short}.py" verify
  cp "scratch/publication_audit_${short}.py" "$post/publication-audit-source.py"
  git add "$audit" "$post" "results/0002-pilot-execution/${group}-20261017-supervision" "results/0002-pilot-predictions/v2-reconstructed/${group}" results/0002-evaluation-completion
  python "scratch/postrun_${short}.py" publication_audit
  git add "$post"
done
