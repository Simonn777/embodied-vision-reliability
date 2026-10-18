"""Hash a hypothetical completion suffix in memory; never restore a missing log."""
import hashlib,json
from pathlib import Path
B=Path('/workspace/scratch/a349d03a0e0a');C=Path(__file__).parent;N=B/'conditional-residual-0006/cases';name='d3-colored-0-blindspot'
def main():
 ex=json.loads((N/(name+'.execution.json')).read_bytes());receipt=N/name/'receipt.json';body=(N/(name+'.stdout.log')).read_bytes()
 rh=hashlib.sha256(receipt.read_bytes()).hexdigest();assert rh==ex['receipt_sha256']
 # The suffix is derived from the frozen case return/print recipe, not recovered historical bytes.
 suffix=(json.dumps({'status':'completed','case':'d3-colored-0-blindspot','receipt_sha256':rh},allow_nan=False)+'\n').encode()
 result={'case':name,'current_stdout_bytes':len(body),'current_stdout_sha256':hashlib.sha256(body).hexdigest(),'recorded_stdout_sha256':ex['stdout_sha256'],'hypothetical_prefix_plus_frozen_completion_sha256':hashlib.sha256(body+suffix).hexdigest(),'hypothetical_suffix_bytes':len(suffix),'matches_recorded_hash':hashlib.sha256(body+suffix).hexdigest()==ex['stdout_sha256'],'actual_case_exit':ex['exit_code'],'receipt_hash_matches':True,'interpretation':'Current stdout lacks the completion line. Appending the derived frozen completion recipe in memory matches the recorded hash if matches_recorded_hash is true. This is not recovery of original log bytes or proof of a filesystem cause. No log or execution receipt was edited. The provenance discrepancy remains open.','no_original_log_modified':True}
 with(C/'log-gap-diagnostic.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
