set -e
for group in g-016 g-032 g-035 g-026
do
  short=${group/-/}
  python scratch/prepare_completion.py "$group"
  python "scratch/supervise_${short}.py"
  audit="results/0002-pilot-execution/${group}-20261017"
  post="results/0002-pilot-execution/${group}-20261017-postrun"
  cp "scratch/verify_${short}.py" "$audit/verification-source.py"
  if [ "$group" = g-016 ]; then
    cp scratch/software_g016.py "$audit/software-checks-source.py"
    python scratch/postrun_g016.py software
  fi
  python "scratch/postrun_${short}.py" verify
  cp "scratch/publication_audit_${short}.py" "$post/publication-audit-source.py"
  git add "$audit" "$post" "results/0002-pilot-execution/${group}-20261017-supervision" "results/0002-pilot-predictions/v2-reconstructed/${group}" results/0002-calibration-completion
  python "scratch/postrun_${short}.py" publication_audit
  git add "$post"
done
