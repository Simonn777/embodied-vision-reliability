# First development configuration: integration provenance

The teacher, training and rollout source were published at 43aef1615ee95d636f5fe9a712740f87ea8b3e17. The generic rollout metric driver and serial continuation were published at 0131186252270c86e36bc5540523ff66f091b7d1, whose CI run 39404457247 passed.

The continuation inherits the historical `expected_commit` field 43aef16 from the teacher outer specification. That field alone is not a claim that the later integration driver existed in that commit. Each operation separately verifies its complete source SHA-256 map, including the later evaluator and the unchanged metric implementation. The continuation itself records its source SHA-256. Both commits and exact source bindings are retained. No operation output or original specification has been rewritten.

The initial continuation was stopped before dispatching any child, solely to add a metric operation immediately after each rollout. Its state and source remain saved. The second continuation runs the complete declared comparison serially, stopping on any failed child.
