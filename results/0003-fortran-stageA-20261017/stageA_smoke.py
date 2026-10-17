import os,sys
# Use existing frozen-version runtime cache only; no pip/network/install.
os.environ['PYTHONPATH']='../bootstapir-cache/runtime-fixed:../bootstapir-cache/runtime:src'
os.execve(sys.executable,[sys.executable,'scripts/run_experiment.py','--smoke','--output','scratch/fortran-stageA-toy-smoke','--protocol','protocols/0001-synthetic-localization.md'],os.environ)
