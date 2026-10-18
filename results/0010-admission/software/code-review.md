# 0010 final CLI/version delta review

**No blocking defect found in the reviewed delta, subject to the documented external admission gates.** The acquisition helper and reused supervisor remain byte-identical to their prior reviewed versions. No acquired source was executed, no network or ML runtime was used, and no repository/source file was edited.

All 23 file size/SHA256 pairs in the final integration receipt match the current files. Helper and fixed-JSON pins match their targets. The portable `HERE.parents[1]` root correctly resolves the repository from `experiments/0010`; the supervisor is loaded from that root's `scripts/`. Before an asset GET, bundled bytes and path separation are checked and the fresh asset root is exclusively reserved. Materialization verifies every input before no-overwrite hard links; existing roots, partials, symlinks, identity mismatches or cross-filesystem hard-link failures stop without fallback. The fixed selection remains bound by its SHA256.

The public caller requires an expected configuration digest and verifies it before child launch. Publication/freeze verification is explicitly an outer admission prerequisite; this CLI does not itself prove that an arbitrary supplied digest was published. Dispatch must supply the reviewed config SHA below from the freeze, not derive an allegedly trusted value from an unverified file. Preparation/probe supervision uses the unchanged helper with 900/300 seconds, fixed four CPUs and 8 GiB limits; the child additionally checks affinity, AS and finite exact cgroup limit.

Fixed manifest caps plus previously charged direct reads sum to exactly 512 MiB. The two subcaps are separate and do not transfer unused allowance. The preserved HEAD record matches the frozen ZIP URL/size/linked SHA and supports the exact additional host `us.aws.cdn.hf.co`; it documents one HEAD with zero body reads, not a successful ZIP GET. New hosts or post-failure retries are not implicitly allowed.

Distribution and module versions are correctly separate: regex's distribution release is pinned while its potentially different `__version__` is retained rather than incorrectly compared to the release number. Pillow distribution mapping is explicit. This check alone is not import-origin authentication; the documented parent-controlled fresh-overlay install/import-origin gate remains required. wcwidth is covered by that dependency gate, not by the eight-module model config. There is no automatic installer in the CLI and no claim otherwise.

Independent delta validation: `PYTHONDONTWRITEBYTECODE=1 python3 -W error -m unittest test_cli test_capability -v` in the integration directory completed with **20 artificial tests passed**, zero failures/skips, 0.043 s unittest time, actual exit 0. Fixtures cover tampered helper/config, pre-network existing-root rejection, portable/no-overwrite materialization, path/hash failures, 900/300-second caller parameters, and distribution/module-version distinction. This is separate from the producer's preserved 54-test receipt and is not a capability result.

Reviewed SHA256:
- CLI: `08f8f918a045dd22855c69049ba1695c98f8df06725ae87fc30d542439c7143e`
- runner: `532ef917a87e976dfd1bc0fd1858c16ea8a4b5bebaa9569add017e394f6a7ba6`
- config: `f4fa4d2645dfbbff5306f2550136f448a4bbc967e0c772b1d99abfef3bfdb8d6`
- integration release receipt: `4b73bbd6ab2625fa592943e9d725080c021fc6aa9858211c7dddbc1b5f3b92c2`

AI-assisted independent software review only. Published freeze, actual offline dependency preparation and first bounded model operation remain distinct future gates.
