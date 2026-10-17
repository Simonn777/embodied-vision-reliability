"""Verify 0003 design freeze only; no dataset loading/model or evaluation modes."""
import argparse
import json
from pathlib import Path
from evr.motion_design import verify_design

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    print(json.dumps(verify_design(a.repo),indent=2))
if __name__=='__main__':main()
