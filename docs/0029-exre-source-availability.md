# ExRE: bounded full-source availability and overlap review

2026-10-20. Source research only. No study outcome, scientific input, model, native call or experiment was read/run. The fixed 0029 study is unchanged. **Outcome: exact ExRE reproduction remains unresolved; this review did not obtain its full method text or an identified official implementation.** The filename records the full-source search, not a claim that the full paper was read.

## Primary sources actually inspected

1. Cho, Kim, Kim, Moon and Kim, *Exploiting recollection effects for memory-based video object segmentation*, Image and Vision Computing 140, 104866, December 2023, DOI [10.1016/j.imavis.2023.104866](https://doi.org/10.1016/j.imavis.2023.104866). The [publisher page](https://www.sciencedirect.com/science/article/pii/S0262885623002408) was available through the web index as abstract, introduction and section snippets; direct opens returned HTTP 403. The complete methods, equations, algorithm and implementation details were **not** returned. No paywall/access control was bypassed.
2. [First-author publication page](https://cekkec.github.io/research/), ExRE entry: paper link leads to the same publisher; no ExRE code or public-manuscript link accompanies that entry. The [author's official lab page](https://ami.khu.ac.kr/people/students/EnkiCho/) links this personal site and lists the paper. The [lab publication list](https://ami.khu.ac.kr/pub/) likewise lists the citation without an ExRE artifact link.
3. [University publication record](https://khu.elsevierpure.com/en/publications/exploiting-recollection-effects-for-memory-based-video-object-seg/): bibliographic record, abstract and DOI link, not an attached manuscript.
4. [Author GitHub profile](https://github.com/cekkec) and its public repository inventory: ten repositories, none identified as ExRE. The visible [STCN fork](https://github.com/cekkec/STCN) describes the original NeurIPS 2021 method; its rendered README contains no ExRE attribution. I did not inspect every fork branch or infer that this fork implements the 2023 paper. Targeted exact-title/DOI, PDF/arXiv, author-code and ETRI searches did not locate a lawful full manuscript or identified implementation. This is a bounded negative search result, not proof that none exists.

Provider-mediated web bytes are unmetered. One direct public GitHub metadata request, `https://api.github.com/users/cekkec/repos?per_page=100`, transferred **50,445 response-body bytes**, SHA256 `189b5ea6ec8e13720cf45556e17b08676d8a32e34795528621d74271d43cfd02`, under a 250,000-byte response cap. No repository, paper PDF, checkpoint or dataset was downloaded. An unrelated/misindexed ResearchGate citation was excluded rather than used to repair the publisher bibliography.

## What is established, and what is not

The publisher's abstract/introduction establishes that ExRE measures prediction stability under presence/absence of memory features. It uses the resulting uncertainty for **both deciding whether to store a current frame and deciding which memory information to remove when capacity is reached**. It presents an integration into existing memory networks rather than merely a segmentation-quality prediction head. Its stated semi-supervised VOS setting supplies first-frame annotations. These facts already occupy generic counterfactual memory influence and selective removal. [Publisher](https://www.sciencedirect.com/science/article/pii/S0262885623002408).

| Required detail for exact control | Actual read status |
|---|---|
| Recollection-effect equation, divergence/norm, spatial/object reduction | Unread; no equation was recovered |
| Sign/orientation of score and normalization | Unread; do not assume unsigned IoU change |
| Which features are omitted, at what time, and how many alternative reads | Unread |
| Admission threshold, eviction ranking/ties, memory capacity/sampling schedule | Unread beyond qualitative admission/eviction purposes |
| Runtime use of initial labels, transferred labels, pseudo-labels or another reference | Not resolved by the first-frame-supervision task description |
| Training/tuning supervision and benchmark-specific parameters | Unread |
| Official reproducible commit/config/checkpoint and license | Not identified |

Consequently, I cannot transcribe an exact ExRE formula, establish that it lacks signed label evidence, or reproduce it faithfully from the sources accessed. The introduction's uncertainty discussion does not determine these implementation choices.

## Narrow relation to frozen 0029

Our checked protocol defines its own observable explicitly: fixed initial-mask labels transferred by SuperPoint/LightGlue; balanced positive/negative softplus loss on the current logits; select D1/D3 only if loss improves over B by more than 1e-6, otherwise B. Only two prescribed old memories are removed in discarded current-state forks, with no future-policy update. These are facts about our frozen protocol, **not evidence that ExRE lacks an equivalent construction**.

The possible distinction to test against a future exact ExRE implementation is label-referenced benefit on the same current-frame counterfactual bank, versus ExRE's actual stability/uncertainty criterion. It remains conditional until its equation and label dependencies are read. Neither “we delete whereas ExRE stores” nor “we measure benefit whereas ExRE only measures influence” is established here: the former conflicts with its introduction, and the latter would require its complete score definition.

For eventual comparison, preserve two separate questions: (a) faithfully reproduce the published memory policy on its supported backbone/configuration; (b) clearly label any adaptation of its score to the fixed SAM2 B/D1/D3 bank. These policies and workloads need not coincide. The current `max_unsigned_change = 1-IoU(mask,B)` control stays an ordinary same-bank control, **not ExRE**. Current diagnostic success, if any, cannot establish superiority over ExRE.

Minimal remaining source requirement: a lawful author manuscript or official code establishing the score equation, reference signal, feature omission and selection/eviction rules, plus parameters. No author contact, paid access or new experiment was initiated. This evidence gap does not alter the frozen diagnostic or authorize post-outcome control tuning.
