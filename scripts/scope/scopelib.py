"""Shared, network-free helpers for the benchmark scripts.

Everything here is pure and deterministic: scope-line parsing, snapshot paths, a light YAML
view of workflows (triggers, local `uses:` references), and JSON output. Both
`scripts/scope/*` (offline scope builders) and `scripts/resources/*` (network fetchers) import
this module; network code lives only in `scripts/resources/ghlib.py`.

Scope-line grammar (one per line, `#` starts a comment line):

    owner/repo                                   repository, commit not pinned yet
    owner/repo@<40-hex sha>                      repository pinned to a commit
    owner/repo@<40-hex sha>:<path>               one workflow file of a pinned repository
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESOURCES = REPO_ROOT / "resources"
REPOS_DIR = RESOURCES / "repos"
SCOPE_DIR = REPO_ROOT / "scope"
OUTPUT_DIR = REPO_ROOT / "output"

# Fixed benchmark parameters (docs/EXPERIMENTS.md). Changing any of them changes the benchmark.
CUTOFF = "2026-09-01T00:00:00Z"
POOL_QUERY = "language:TypeScript stars:>2500 fork:false archived:false"
POOL_SORT = "stars"
POOL_ORDER = "desc"
POOL_SIZE = 300
SEED = 42
EVENTS = ("push", "pull_request")

WORKFLOWS_DIR = ".github/workflows"
ROOT_FILES = ("package.json", "tsconfig.json", "tsconfig.build.json")
ACTION_FILES = ("action.yml", "action.yaml")

_LINE_RE = re.compile(
    r"(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    r"(?:@(?P<sha>[0-9a-f]{40}))?"
    r"(?::(?P<path>\S+))?"
)


# ---------------------------------------------------------------------------
# Scope lines and files
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class ScopeLine:
    repo: str
    sha: str | None = None
    path: str | None = None

    @staticmethod
    def parse(text: str) -> "ScopeLine":
        m = _LINE_RE.fullmatch(text.strip())
        if m is None:
            raise ValueError(f"malformed scope line: {text!r}")
        return ScopeLine(m.group("repo"), m.group("sha"), m.group("path"))

    def __str__(self) -> str:
        s = self.repo
        if self.sha is not None:
            s += f"@{self.sha}"
        if self.path is not None:
            s += f":{self.path}"
        return s

    @property
    def owner(self) -> str:
        return self.repo.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.repo.split("/", 1)[1]

    def benchmark_id(self) -> str:
        """`<owner>__<repo>__<workflow-file-stem>` (docs/ARCHITECTURE.md)."""
        if self.path is None:
            raise ValueError(f"not a workflow line: {self}")
        stem = posixpath.splitext(posixpath.basename(self.path))[0]
        return f"{self.owner}__{self.name}__{stem}"


def read_scope(path: Path) -> list[ScopeLine]:
    lines: list[ScopeLine] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        lines.append(ScopeLine.parse(text))
    return lines


def format_scope(lines: Iterable[ScopeLine | str]) -> str:
    """Sorted, de-duplicated scope text with a trailing newline (empty scope -> empty file)."""
    unique = sorted({str(x) for x in lines})
    return "".join(f"{s}\n" for s in unique)


def write_scope(path: Path, lines: Iterable[ScopeLine | str]) -> int:
    text = format_scope(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text.count("\n")


# ---------------------------------------------------------------------------
# Snapshot layout
# ---------------------------------------------------------------------------


def repo_dirname(repo: str) -> str:
    owner, name = repo.split("/", 1)
    return f"{owner}__{name}"


def snapshot_dir(repo: str, sha: str, repos_dir: Path = REPOS_DIR) -> Path:
    return repos_dir / repo_dirname(repo) / sha


def load_manifest(repo: str, sha: str, repos_dir: Path = REPOS_DIR) -> dict | None:
    p = snapshot_dir(repo, sha, repos_dir) / "manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def is_complete(repo: str, sha: str, repos_dir: Path = REPOS_DIR) -> bool:
    m = load_manifest(repo, sha, repos_dir)
    return bool(m and m.get("complete"))


def safe_relpath(path: str) -> str | None:
    """Normalize a repository-relative path; None if it escapes the repository root."""
    norm = posixpath.normpath(path.strip())
    if norm in ("", "."):
        return ""
    if norm.startswith("../") or norm == ".." or norm.startswith("/"):
        return None
    return norm


def is_workflow_file(path: str) -> bool:
    """Files GitHub runs as workflows: directly inside .github/workflows, .yml or .yaml."""
    return posixpath.dirname(path) == WORKFLOWS_DIR and path.endswith((".yml", ".yaml"))


# ---------------------------------------------------------------------------
# Light YAML view
# ---------------------------------------------------------------------------


def load_yaml(text: str) -> Any:
    """`yaml.safe_load`, returning None on any YAML error."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


def workflow_triggers(doc: Any) -> frozenset[str]:
    """Event names in a workflow's `on:` block.

    PyYAML (YAML 1.1) reads the bare key `on` as the boolean True, so both keys are checked.
    """
    if not isinstance(doc, dict):
        return frozenset()
    on = doc.get("on", doc.get(True))
    if isinstance(on, str):
        return frozenset({on})
    if isinstance(on, list):
        return frozenset(x for x in on if isinstance(x, str))
    if isinstance(on, dict):
        return frozenset(k for k in on if isinstance(k, str))
    return frozenset()


def triggers_core_event(doc: Any) -> bool:
    """True if the workflow runs on `push` or `pull_request` (the GHA Core events)."""
    return bool(workflow_triggers(doc) & set(EVENTS))


def normalize_local_ref(uses: str) -> str | None:
    """`./path` -> repository-relative directory ('' for the root); None if not local/resolvable."""
    if not isinstance(uses, str):
        return None
    u = uses.strip()
    if not u.startswith("./") and u != ".":
        return None
    if "${{" in u:
        return None
    return safe_relpath(u)


def _step_uses(steps: Any) -> list[str]:
    out: list[str] = []
    if isinstance(steps, list):
        for s in steps:
            if isinstance(s, dict) and isinstance(s.get("uses"), str):
                out.append(s["uses"])
    return out


@dataclass(frozen=True)
class LocalRefs:
    actions: tuple[str, ...] = ()      # normalized directories of local actions (`uses: ./dir`)
    workflows: tuple[str, ...] = ()    # normalized paths of local reusable workflows
    unresolved: tuple[str, ...] = ()   # local `uses:` strings that cannot be resolved statically


def local_refs_in_workflow(doc: Any) -> LocalRefs:
    actions: set[str] = set()
    workflows: set[str] = set()
    unresolved: set[str] = set()
    jobs = doc.get("jobs") if isinstance(doc, dict) else None
    if isinstance(jobs, dict):
        for key in sorted(jobs, key=str):
            job = jobs[key]
            if not isinstance(job, dict):
                continue
            u = job.get("uses")
            if isinstance(u, str) and u.strip().startswith("./"):
                n = normalize_local_ref(u)
                (workflows.add(n) if n else unresolved.add(u.strip()))
            for su in _step_uses(job.get("steps")):
                if su.strip().startswith("./") or su.strip() == ".":
                    n = normalize_local_ref(su)
                    (actions.add(n) if n is not None else unresolved.add(su.strip()))
    return LocalRefs(tuple(sorted(actions)), tuple(sorted(workflows)), tuple(sorted(unresolved)))


def local_refs_in_action(doc: Any) -> LocalRefs:
    """Local actions referenced by a composite action's `runs.steps`."""
    actions: set[str] = set()
    unresolved: set[str] = set()
    runs = doc.get("runs") if isinstance(doc, dict) else None
    if isinstance(runs, dict):
        for su in _step_uses(runs.get("steps")):
            if su.strip().startswith("./") or su.strip() == ".":
                n = normalize_local_ref(su)
                (actions.add(n) if n is not None else unresolved.add(su.strip()))
    return LocalRefs(tuple(sorted(actions)), (), tuple(sorted(unresolved)))


def action_file_candidates(action_dir: str) -> tuple[str, ...]:
    return tuple(posixpath.join(action_dir, f) if action_dir else f for f in ACTION_FILES)


def find_action_file(snap: Path, action_dir: str) -> str | None:
    for rel in action_file_candidates(action_dir):
        if (snap / rel).is_file():
            return rel
    return None


def local_closure(snap: Path, workflow_path: str) -> tuple[frozenset[str], frozenset[str]]:
    """Offline transitive closure of a workflow's local references inside a snapshot.

    Returns (action directories, reusable workflow paths), including references reached through
    reusable workflows and composite actions. Missing files are skipped.
    """
    actions: set[str] = set()
    workflows: set[str] = set()
    todo_wf = [workflow_path]
    seen_wf: set[str] = set()
    while todo_wf:
        wf = todo_wf.pop()
        if wf in seen_wf:
            continue
        seen_wf.add(wf)
        p = snap / wf
        if not p.is_file():
            continue
        refs = local_refs_in_workflow(load_yaml(p.read_text(encoding="utf-8")))
        for w in refs.workflows:
            workflows.add(w)
            todo_wf.append(w)
        actions.update(refs.actions)
    todo_act = sorted(actions)
    while todo_act:
        d = todo_act.pop()
        rel = find_action_file(snap, d)
        if rel is None:
            continue
        for a in local_refs_in_action(load_yaml((snap / rel).read_text(encoding="utf-8"))).actions:
            if a not in actions:
                actions.add(a)
                todo_act.append(a)
    return frozenset(actions), frozenset(workflows)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(obj), encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_oid(data: bytes) -> str:
    """Git's object id of a blob with this content (used to verify fetched files)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Workflow statistics (eval/random scopes); provided by ghav.pipeline
# ---------------------------------------------------------------------------

STATS_KEYS = ("instances", "frozen", "redundancy_ratio")


def load_workflow_stats():
    """`ghav.pipeline.workflow_stats(benchmark_line) -> {instances, frozen, redundancy_ratio}`.

    Imported lazily so the other scope builders work before the pipeline exists.
    """
    try:
        from ghav.pipeline import workflow_stats  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(
            "error: this scope needs ghav.pipeline.workflow_stats(benchmark_line) -> "
            f"dict(instances, frozen, redundancy_ratio), which is not available yet ({e})") from e
    return workflow_stats


def validate_stats(st: Any) -> dict:
    if not isinstance(st, dict) or any(k not in st for k in STATS_KEYS):
        raise ValueError(f"workflow_stats must return keys {STATS_KEYS}, got {st!r}")
    inst, frozen, ratio = st["instances"], st["frozen"], st["redundancy_ratio"]
    if isinstance(inst, bool) or not isinstance(inst, int) or isinstance(frozen, bool) \
            or not isinstance(frozen, int) or not 0 <= frozen <= inst:
        raise ValueError(f"instances/frozen must be counts with 0 <= frozen <= instances: {st!r}")
    if not isinstance(ratio, (int, float)) or not 0.0 <= float(ratio) <= 1.0:
        raise ValueError(f"redundancy_ratio must be in [0, 1]: {st!r}")
    return {"instances": inst, "frozen": frozen, "redundancy_ratio": float(ratio)}


def collect_stats(lines: Iterable[ScopeLine], fn) -> tuple[dict[str, dict], dict[str, str]]:
    """Stats per workflow line in sorted order; failures (e.g. lowering errors) are recorded."""
    stats: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for ln in sorted(lines):
        key = str(ln)
        try:
            stats[key] = validate_stats(fn(key))
        except Exception as e:  # noqa: BLE001 - any failure excludes the workflow and is reported
            errors[key] = f"{type(e).__name__}: {e}"
    return stats, errors
