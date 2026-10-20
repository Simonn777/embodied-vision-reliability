"""Independent text/scalar publication verification; never imports study code."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

P = Path(__file__).resolve().parents[2]
R = P / "work/embodied-vision-reliability"
PREP = P / "work/0029-preparation"
PACKAGE = PREP / "native-attempt02-publication-01"
RUN = P / "work/0029-native-development-02"
OUT = PREP / "attempt02-publication-review.json"
COMMIT = "098dee64878b5ef33edbac12223f86da176e091e"
COHORT = (("dogs-scale",83), ("upside-down",65), ("dog-agility",25),
          ("cat-girl",89), ("planes-water",38), ("dog-gooses",86))
METHODS = ("signed_deletion","baseline","max_predicted_iou","max_unsigned_change",
           "always_D1","always_D3","random_deletion","random_on_signed","native_bank",
           "prompt_direct","prompt_signed","reset_direct","reset_signed")
ALIASES = ((str(R),"$REPOSITORY"),(str(R.parent),"$PRIVATE"),
           (str(P),"$PROJECT"),(str(Path.home()),"$USER_HOME"))
errors, notes, originals = [], [], {}
checks = 0


def check(ok, name, detail=None):
    global checks
    checks += 1
    if not ok:
        errors.append({"check":name, "detail":detail})


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def text_bytes(path):
    path = Path(path)
    if path.suffix not in (".json",".csv",".log",".md",".py"):
        raise ValueError("Non-text input forbidden: "+str(path))
    raw = path.read_bytes()
    raw.decode("utf-8")
    return raw


def load(path):
    return json.loads(text_bytes(path), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def redact(raw):
    text = raw.decode()
    for prefix, alias in ALIASES:
        text = text.replace(prefix, alias)
    return text.encode()


def origin_path(value):
    for prefix, alias in ALIASES:
        if value == alias or value.startswith(alias+"/"):
            return Path(prefix+value[len(alias):])
    raise ValueError("Unrecognized origin alias")


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def cell(value):
    return "" if value is None else str(value)


def table(name, expected):
    with (PACKAGE/name).open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        keys = reader.fieldnames
    check(len(rows)==len(expected),name+".count",{"actual":len(rows),"expected":len(expected)})
    check(keys==list(expected[0]),name+".columns")
    for i,(a,b) in enumerate(zip(rows,expected)):
        for key,value in b.items():
            check(a.get(key)==cell(value),f"{name}.{i}.{key}",
                  None if a.get(key)==cell(value) else {"actual":a.get(key),"expected":cell(value)})
    return rows


manifest_raw = text_bytes(PACKAGE/"publication-manifest.json")
manifest = json.loads(manifest_raw)
check(digest(manifest_raw)=="201eb7e8165f27c7f4cad7e9e18534509ead7b644661e6fe452592d8ed93c319",
      "reported package manifest hash")
check(manifest["source_commit"]==COMMIT and manifest["completed"] is True,"package completion metadata")
entries = manifest["files"]
names = [e["path"] for e in entries]
actual = [p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*") if p.is_file()]
check(len(names)==len(set(names))==74,"74 unique listed artifacts")
check(set(actual)==set(names)|{"publication-manifest.json"},"exact75 text artifacts")
for entry in entries:
    name = entry["path"]
    path = PACKAGE/name
    check(not path.is_symlink() and ".." not in Path(name).parts,"safe public path."+name)
    raw = text_bytes(path)
    check(digest(raw)==entry["public_sha256"] and len(raw)==entry["public_bytes"],"public bytes."+name)
    check(entry["omissions"]==[],"declared omissions."+name)
    for source in entry["origins"]:
        original = origin_path(source["origin"])
        value = text_bytes(original)
        originals[str(original)] = digest(value)
        check(digest(value)==source["original_sha256"] and len(value)==source["original_bytes"],
              "original bytes."+name+"."+source["origin"])
    transformation = entry["transformation"]
    if transformation.startswith("Byte-identical"):
        check(raw==redact(text_bytes(origin_path(entry["origins"][0]["origin"]))),
              "literal redaction reconstruction."+name)
    elif transformation.startswith("Complete indicated JSON subtree"):
        source = entry["origins"][0]
        value = load(origin_path(source["origin"]))
        for key in source["json_pointer"].split("/")[1:]:
            value = value[key.replace("~1","/").replace("~0","~")]
        check(raw==redact(json_bytes(value)),"declared subtree reconstruction."+name,
              "Declared JSON pointer must reproduce published subtree; constructed summaries require their own derivation.")
    elif name not in {"README.md","operations.csv","methods.csv","sequences.csv","queries.csv",
                      "gate.csv","headroom.csv","query-cost-columns.csv"}:
        check(False,"unrecognized derivation."+name,transformation)

for name in actual:
    raw = text_bytes(PACKAGE/name)
    check(b"\0" not in raw,"no binary/NUL."+name)
    for marker in (b"/Users/",b"/home/",b"/private/var/",b"/var/folders/"):
        check(marker not in raw,"no private path."+name+"."+marker.decode())
    for pattern in (rb"gh[pousr]_[A-Za-z0-9]{30,}", rb"sk-[A-Za-z0-9_-]{30,}",
                    rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"):
        check(re.search(pattern,raw) is None,"no credential pattern."+name)

record = load(RUN/"evaluation/receipt.json")
e = record["evaluation"]
s = e["summary"]
audit = load(RUN/"audit-01/receipt.json")
check(record["status"]=="completed" and record["eligible"] and record["query_masks_decoded"]==48,
      "complete evaluator")
check(audit["status"]=="completed" and audit["eligible"] and audit["errors"]==[]
      and audit["checks"]==12463 and audit["query_masks_decoded"]==48,"complete independent audit")
check(audit["independent_gate"]==e["gate"],"audit/evaluator gate equality")
identities = [(n,4+(2*j*(count-5)+7)//14) for n,count in COHORT for j in range(8)]
check([(x["sequence"],x["frame"]) for x in e["rows"]]==identities,"48 fixed evaluation identities")
check([(x["sequence"],x["frame"]) for x in s["rows"]]==identities,"48 fixed summary identities")
check(all(x["status"]=="completed" and set(x["selected_branches"])==set(METHODS) for x in e["rows"]),
      "13 methods every query")

table("methods.csv",[dict(method=m,pooled_j=s["pooled"][m],
    **{n:s["videos"][n]["means"][m] for n,_ in COHORT}) for m in METHODS])
table("sequences.csv",[dict(sequence=n,frame_count=count,query_count=8,
    actionable_queries=e["coverage"]["actionable_queries_by_video"][n],
    signed_j=s["videos"][n]["means"]["signed_deletion"],baseline_j=s["videos"][n]["means"]["baseline"],
    best_control=s["videos"][n]["best_control"],
    best_control_j=s["videos"][n]["means"][s["videos"][n]["best_control"]],
    signed_margin=s["videos"][n]["signed_margin"],positive=s["videos"][n]["signed_margin"]>0,
    above_minus_002=s["videos"][n]["signed_margin"]>=-.02) for n,count in COHORT])
queries=[]
for row,metric in zip(e["rows"],s["rows"]):
    q={k:row[k] for k in ("sequence","frame","target_id","status","actionable")}
    for m in METHODS:
        q[m+"_branch"]=row["selected_branches"][m]
        q[m+"_j"]=metric["j"][m]
    for b in ("B","D1","D3","M0","M1","M2","P","R"):
        q[b+"_candidate_j"]=row["candidate_j"].get(b)
    q.update(signed_j_delta=row["signed_harm"]["j_delta"],signed_drop_gt_005=row["signed_harm"]["drop_gt_005"])
    queries.append(q)
table("queries.csv",queries)
gates=[dict(clause="coverage",value=e["coverage"]["videos_with_evidence"],threshold=4,
            relation=">=",passed=e["gate"]["coverage"])]
for clause,key,threshold in (("candidate_information","candidate_gain",.02),
        ("pooled_selection","signed_gain",.02),("four_positive_videos","positive_videos",4),
        ("no_video_below_minus_002","worst_video_margin",-.02)):
    gates.append(dict(clause=clause,value=s[key],threshold=threshold,relation=">=",passed=e["gate"][clause]))
table("gate.csv",gates)
check(all(r["passed"]==(r["value"]>=r["threshold"]) for r in gates),"saved scalar gate decisions")
check(e["gate"]["pass"]==all(r["passed"] for r in gates)==False,"overall frozen gate false")
headroom=[]
for scope in ("pooled",*(n for n,_ in COHORT)):
    v=s["pooled"] if scope=="pooled" else s["videos"][scope]["means"]
    d=e["diagnostics"]["pooled"] if scope=="pooled" else e["diagnostics"]["videos"][scope]
    headroom.append(dict(scope=scope,baseline_j=v["baseline"],deletion_admissible_j=v["oracle_admissible"],
        deletion_unrestricted_j=v["oracle_unrestricted"],ungated_D1_j=v["ungated_D1"],
        ungated_D3_j=v["ungated_D3"],**d["native_bank"]))
table("headroom.csv",headroom)
table("query-cost-columns.csv",[dict(unit=u,column=n,sum=v["sum"],reported_queries=v["reported_queries"])
    for u,columns in e["diagnostics"]["pooled"]["costs"].items() for n,v in columns.items()])

ref=load(PACKAGE/"attempt01-reference.json")
old=R/ref["public_directory"]
oldmanifest=load(old/"publication-manifest.json")
check(digest(text_bytes(old/"publication-manifest.json"))==ref["manifest_sha256"],"historical manifest binding")
oldfiles={x["path"]:x for x in oldmanifest["files"]}
operations=[]
for n,_ in COHORT:
    relative=n+"/supervision/receipt.json"
    if relative not in oldfiles:
        check(n in ("planes-water","dog-gooses"),"allowed old unrun identity."+n)
        operations.append(dict(attempt="01",phase="video/"+n,status="not_run",source_commit=None,
            exit_code=None,termination=None,wall_seconds=None,peak_group_rss_kib=None,spec_sha256=None,
            outer_sha256=None,artifact_sha256=None,cost_scope="not executed; no zero cost invented"))
        continue
    b=text_bytes(old/relative); o=json.loads(b)
    check(digest(b)==oldfiles[relative]["public_sha256"],"old published receipt hash."+n)
    operations.append(dict(attempt="01",phase="video/"+n,status=o["status"],source_commit=o["expected_commit"],
        exit_code=o["exit_code"],termination=o.get("termination"),wall_seconds=o["wall_seconds"],
        peak_group_rss_kib=o["peak_group_rss_kib"],spec_sha256=o["spec_sha256"],
        outer_sha256=oldfiles[relative]["original_sha256"],artifact_sha256=o.get("output_sha256"),
        cost_scope="historical disjoint actual outer; no efficacy admission"))
check(ref["status"]=="technically_ineligible" and ref["operations"]==operations,
      "historical summary values independently reconstruct")
check(operations[3]["status"]=="failed" and operations[3]["termination"]=="descendants_remain",
      "old cat-girl failure retained")
new_video_seconds=[]
for phase,specpath,receipt in [
    *(("videos/"+n,RUN/n/"spec.json",RUN/n/"video/receipt.json") for n,_ in COHORT),
    ("gt",RUN/"gt-spec.json",RUN/"gt/receipt.json"),
    ("evaluation",RUN/"evaluation-spec.json",RUN/"evaluation/receipt.json"),
    ("audit",RUN/"audit-spec-01.json",RUN/"audit-01/receipt.json")]:
    spec=load(specpath); outerpath=Path(spec["operation_output"])/"receipt.json"; o=load(outerpath)
    check(o["status"]=="completed" and o["exit_code"]==0 and o["termination"] is None
          and o["remaining_descendant_pids"]==[] and o["cleanup_errors"]==[] and o["signal_attempts"]==[],
          "clean new outer."+phase)
    check(o["expected_commit"]==spec["commit"]==COMMIT and o["spec_sha256"]==digest(text_bytes(specpath))
          and o["output_sha256"]==digest(text_bytes(Path(spec["result_file"]))),"new outer hashes."+phase)
    check(all(o["log_sha256"][n]==digest(text_bytes(outerpath.parent/n))
              for n in ("stdout.log","stderr.log")),"new outer logs."+phase)
    check(o["post_exit_snapshot"]["rows"]==[] and o["cleanup_snapshots"][-1]["rows"]==[],
          "new clean postexit observations."+phase)
    operations.append(dict(attempt="02",phase=phase,status=o["status"],source_commit=o["expected_commit"],
        exit_code=o["exit_code"],termination=o.get("termination"),wall_seconds=o["wall_seconds"],
        peak_group_rss_kib=o["peak_group_rss_kib"],spec_sha256=digest(text_bytes(specpath)),
        outer_sha256=digest(text_bytes(outerpath)),artifact_sha256=digest(text_bytes(receipt)),
        cost_scope="disjoint actual outer operation"))
    if phase.startswith("videos/"):
        new_video_seconds.append(o["wall_seconds"])
        seal=load(Path(spec["result_file"])); video=load(receipt)
        check(seal["receipt_sha256"]==digest(text_bytes(receipt)) and video["query_count"]==8
              and video["sam_sequence_invocations"]==1,"new video seal/coverage."+phase)
table("operations.csv",operations)
check(len(operations)==15 and sum(r["status"]=="not_run" for r in operations)==2,"old/new complete cost population")
readme=text_bytes(PACKAGE/"README.md").decode()
for claim in ("12463 checks,0errors","**FAIL**","all48 fixed queries","all13 methods","0/6",
              "custom fixed-query foreground J","Attempt01 remains technically ineligible",
              "not official DAVIS J&F",f"{100*s['signed_gain']:.9f}",
              f"{100*s['candidate_gain']:.9f}",f"{100*s['worst_video_margin']:.9f}",COMMIT):
    check(claim in readme,"README claim."+claim)
check("novelty" in readme and "does not establish" in readme,"no unsupported efficacy/novelty scope")
for path,expected in originals.items():
    check(digest(text_bytes(path))==expected,"original unchanged."+str(path))
report=dict(schema="evr0029.publication-review.v1",status="PASS" if not errors else "FAIL",
    checks=checks,issues=errors,notes=notes,package_manifest_sha256=digest(manifest_raw),
    script_sha256=digest(text_bytes(__file__)),listed_files=len(entries),total_public_files=len(actual),
    original_text_files_checked=len(originals),csv_counts=dict(methods=13,sequences=6,queries=48,
      deployed_outcomes=624,gates=5,headroom=7,operations=15),
    gate=e["gate"],signed_gain=s["signed_gain"],candidate_gain=s["candidate_gain"],
    old_executed_video_seconds=math.fsum(r["wall_seconds"] for r in operations if r["attempt"]=="01" and r["wall_seconds"] is not None),
    new_video_seconds=math.fsum(new_video_seconds),
    all_listed_executed_outer_seconds=math.fsum(r["wall_seconds"] for r in operations if r["wall_seconds"] is not None),
    scope="Text/scalar reconstruction only; no study imports, NPZ, GT image decode, model or numerical pipeline rerun.")
with OUT.open("x") as f:
    json.dump(report,f,indent=2,allow_nan=False); f.write("\n")
print(json.dumps(report,indent=2))
sys.exit(0 if not errors else 1)
