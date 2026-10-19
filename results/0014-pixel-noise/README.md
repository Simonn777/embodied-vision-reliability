# 0014 first attempt: numerical guard stopped execution

After published freeze32f3e40, pair0's native continuation completed; the N kernel then raised `nonfinite or collapsed residual arithmetic` during its initial evaluation. The worker and supervisor exited1, and the seal is incomplete/ineligible. The saved native call reports zero iterations and unchanged objective; N/P produced no completed pose, no0014 scorer ran, and no other pair was attempted.

Original files and hashes are preserved here. This is not a negative or positive test of pixel-noise accuracy. Diagnosis must first distinguish actual invalid arithmetic from a backend exception using artificial inputs; no numerical guard may be disabled simply to continue. Any technical retry requires a separately recorded freeze and output directory, with all extra native calls counted.
