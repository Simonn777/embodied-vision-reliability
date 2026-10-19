# 0016 result publication verification

Verified2026-10-19. [PR45](https://github.com/Simonn777/embodied-vision-reliability/pull/45)
merged at19:56:22UTC. Reviewed result head
`7afed375db481e0e001c41abc8b18e7f0a93552e` equals the observed pushed remote head.
Merge commit is `748215ca30ac60514726aca310c08b7712f9470d`.

- Scientific freeze `055ea70bdfba2a7a3c284063ab91dd1ecb624c27` preceded numerical
  inputs; [CI39312932830](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39312932830) succeeded.
- Result-head [CI39314566982](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39314566982)
  succeeded for exactly7afed375 before merge.
- Main [CI39315128349](https://github.com/Simonn777/embodied-vision-reliability/actions/runs/39315128349)
  succeeded for exactly748215ca after merge.

The parent observed tool-reported pushed state, merge state and both exact CI
conclusions. All192workers/768calls and the independent saved-output review are
published in [complete evidence](../results/0016-search/README.md). Current
normalized selector AUC5=78.2293102381%; its oracle=78.2587614824%.
The0.0294512443pp necessary headroom gate fails. Neither publication nor passing
CI is an accuracy advance or completion of the broader research goal.

No scientific run was repeated. Original research sources/results and
held-out0022 numerical arrays remain unchanged. A PR-body patch helper initially
rejected a duplicate target operation before changing files; a normal text write
then updated that description. Git's large fast-forward listing was truncated by
the terminal transport; this does not affect the complete scientific receipts.
