#!/usr/bin/env python3
"""Trace the filesystem effects of one job instance of a benchmark workflow on a local machine.

The trace is evidence for observed contracts (docs/CONTRACT_GUIDE.md): for every `run:` step it
records which workspace paths the step created, modified, and deleted.

Procedure (repository-independent):

 1. Lower the workflow from the cached snapshot (resources/repos/...) and select the job
    instance given by `--instance` (a matrix assignment `k=v,...` or an instance id).
    Step conditions are evaluated concretely for `--event` with every task succeeding
    (`ghav.cost.simulate`), exactly as the cost model does.
 2. Shallow-clone the repository at the pinned sha into a fresh temporary directory that must
    lie outside every git repository. The clone stands in for `actions/checkout`.
 3. Package manager: if the root package.json has `packageManager: <pm>@<version>`, a shim on
    PATH runs `npx --yes <pm>@<version>` (this stands in for pnpm/action-setup and corepack).
    Node is the host's node (stands in for actions/setup-node; caches are cold).
 4. Execute only the `run:` steps whose condition holds, in order, with GitHub's default shell
    (`bash --noprofile --norc -eo pipefail`), the workflow/job/step env of the instance,
    CI=true and GITHUB_ACTIONS=true, and an isolated HOME and TMPDIR. `uses:` steps are not
    executed; their known effects are documented in the trace (KNOWN_ACTION_EFFECTS).
 5. Before and after each executed step, take a content-hash snapshot of the workspace
    (sha256 of every file, symlink targets for links; `.git` excluded) and record the diff.

Output: resources/traces/<owner>__<repo>/<sha>/<benchmark-id>.json in the `ghav-trace/1` format
read by `ghav.contracts.load_observed` (docs/CONTRACT_GUIDE.md, observed contracts):

  {"format": "ghav-trace/1", "benchmark": "<scope line>",
   "steps": [{"binding": {"kind", "text", "with", "env", "working_directory", "shell"},
              "writes": [paths], "deletes": [paths], "external": [], "instance": id, "index": i}],
   "instances": {"<instance id>": {"steps": [per-step record, including skipped steps], ...}}}

`steps` lists only `run:` steps that executed and exited 0, over every traced instance (the
loader unions observations of one binding). `writes` = created or modified paths; `deletes` =
removed paths. A filesystem trace cannot see network effects, so `external` is always empty.
`instances` keeps the full per-instance record (skipped actions with their known effects,
conditions, exit codes, `created`/`modified` split, counts). Re-running for another instance
adds it; the same instance is replaced. Paths are workspace-relative POSIX paths; a path ending
in `/` stands for a whole directory that did not exist before the step (in `writes`) or no
longer exists after it (in `deletes`). Wall-clock seconds are reported only; nothing depends
on them.

Usage:
  uv run python scripts/traces/trace_workflow.py \\
      --benchmark 'owner/repo@<sha>:.github/workflows/x.yml' --instance 'build=cjs,env=development'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "scope")]

from scopelib import REPOS_DIR, RESOURCES, ScopeLine, dumps_json, repo_dirname, snapshot_dir  # noqa: E402

TRACES_DIR = RESOURCES / "traces"
FORMAT = "ghav-trace/1"   # must equal ghav.contracts.TRACE_FORMAT
DEFAULT_SHELL = ("bash", "--noprofile", "--norc", "-eo", "pipefail")
_EXPR = re.compile(r"\$\{\{.*?\}\}", re.S)

# What a skipped `uses:` step would have done, and what stands in for it in the trace.
KNOWN_ACTION_EFFECTS: tuple[tuple[str, str], ...] = (
    ("actions/checkout@", "Writes the repository at the pinned commit into the workspace (src). "
                          "Trace: done by the tool's shallow clone before the first snapshot; .git is not snapshotted."),
    ("pnpm/action-setup@", "Installs pnpm (version from packageManager when not given) outside the workspace (dep). "
                           "Trace: PATH shim running `npx --yes pnpm@<packageManager version>`."),
    ("actions/setup-node@", "Installs node outside the workspace and adds it to PATH (dep); with `cache:` restores "
                            "and saves the package-manager store outside the workspace. "
                            "Trace: host node; cold cache (no restore)."),
    ("actions/setup-python@", "Installs python outside the workspace (dep). Trace: host python."),
    ("oven-sh/setup-bun@", "Installs bun outside the workspace (dep). Trace: not provided."),
    ("actions/cache@", "Restores/saves a cache; cold-cache semantics (no effect). Trace: skipped."),
    ("actions/cache/restore@", "Restores a cache; cold-cache semantics (no effect). Trace: skipped."),
    ("actions/cache/save@", "Saves a cache; cold-cache semantics (no effect). Trace: skipped."),
    ("actions/upload-artifact@", "Reads the given paths and writes a run artifact (run:<name>); no workspace write."),
    ("actions/download-artifact@", "Writes the artifact's files into the given path (out). Trace: skipped."),
)
PM_NAMES = ("npm", "pnpm", "yarn")


# ---------------------------------------------------------------------------
# Pure helpers (tested offline)
# ---------------------------------------------------------------------------


def known_effects(uses: str) -> str:
    for prefix, text in KNOWN_ACTION_EFFECTS:
        if uses.startswith(prefix):
            return text
    return "Unknown action: not executed; its effects are not observed by this trace."


def parse_instance(spec: str) -> tuple[str | None, tuple[tuple[str, str], ...]]:
    """`k=v,k2=v2` -> (None, sorted pairs); an instance id `job[k=v,...]` -> (id, ())."""
    s = spec.strip()
    if "[" in s or "=" not in s:
        return s, ()
    pairs = []
    for part in s.split(","):
        if "=" not in part:
            raise ValueError(f"bad matrix assignment: {part!r}")
        k, v = part.split("=", 1)
        pairs.append((k.strip(), v.strip()))
    return None, tuple(sorted(pairs))


def select_instance(program: Any, spec: str, job_key: str | None = None) -> Any:
    iid, pairs = parse_instance(spec)
    jobs = [j for j in program.jobs if job_key is None or j.orig_job == job_key]
    cands = [j for j in jobs if j.id == iid] if iid is not None else [j for j in jobs if tuple(j.matrix) == pairs]
    if len(cands) != 1:
        raise SystemExit(f"error: --instance {spec!r} selects {len(cands)} job instances "
                         f"(available: {', '.join(j.id for j in program.jobs)})")
    return cands[0]


def package_manager(package_json: Path) -> tuple[str, str] | None:
    """(`pnpm`, `11.3.0`) from `packageManager: pnpm@11.3.0+sha512...`, else None."""
    try:
        doc = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    spec = doc.get("packageManager") if isinstance(doc, dict) else None
    if not isinstance(spec, str) or "@" not in spec:
        return None
    name, version = spec.split("@", 1)
    version = version.split("+", 1)[0]
    if name not in PM_NAMES or not re.fullmatch(r"[0-9A-Za-z.\-]+", version):
        return None
    return name, version


def file_digest(path: Path) -> str:
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode):
        return "link:" + os.readlink(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Snapshotter:
    """Content-hash snapshots of a directory tree (excluding `.git`), re-hashing only files
    whose stat signature changed since the previous snapshot."""

    def __init__(self, root: Path, exclude: Iterable[str] = (".git",)):
        self.root = root
        self.exclude = frozenset(exclude)
        self._cache: dict[str, tuple[tuple, str]] = {}

    def snapshot(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in self.exclude)
            rel_dir = os.path.relpath(dirpath, self.root)
            entries = list(filenames) + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]
            for name in sorted(entries):
                full = os.path.join(dirpath, name)
                rel = name if rel_dir == "." else f"{rel_dir}/{name}"
                rel = rel.replace(os.sep, "/")
                try:
                    st = os.lstat(full)
                except FileNotFoundError:
                    continue
                if not (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)):
                    continue
                sig = (st.st_size, st.st_mtime_ns, st.st_ino, st.st_mode)
                hit = self._cache.get(rel)
                if hit is not None and hit[0] == sig:
                    out[rel] = hit[1]
                    continue
                try:
                    d = file_digest(Path(full))
                except (FileNotFoundError, PermissionError):
                    continue
                self._cache[rel] = (sig, d)
                out[rel] = d
            # symlinked directories are recorded above and never descended into
            dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
        return out


def _ancestors(path: str) -> list[str]:
    parts = path.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts))]


def collapse(paths: Iterable[str], other: Mapping[str, str]) -> list[str]:
    """Replace every file under a top-most directory that has no file in `other` by `dir/`."""
    other_dirs: set[str] = set()
    for p in other:
        other_dirs.update(_ancestors(p))
    out: set[str] = set()
    for p in paths:
        top = next((a for a in _ancestors(p) if a not in other_dirs), None)
        out.add(f"{top}/" if top is not None else p)
    return sorted(out)


def diff(before: Mapping[str, str], after: Mapping[str, str]) -> dict[str, Any]:
    created = sorted(p for p in after if p not in before)
    deleted = sorted(p for p in before if p not in after)
    modified = sorted(p for p in after if p in before and before[p] != after[p])
    created_c = collapse(created, before)
    return {
        "created": created_c,
        "modified": modified,
        "deletes": collapse(deleted, after),
        "writes": sorted(set(created_c) | set(modified)),
        "counts": {"created": len(created), "modified": len(modified), "deleted": len(deleted)},
    }


def binding_json(binding: Any) -> dict:
    """The binding as the trace format stores it; `ghav.contracts` rebuilds `Binding.key()`."""
    kind, text, with_, env, wd, shell = binding.key()
    return {"kind": kind, "text": text, "with": {k: v for k, v in with_}, "env": {k: v for k, v in env},
            "working_directory": wd, "shell": shell}


def loader_steps(instances: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    """The flat `steps` list: executed `run:` steps that exited 0, in (instance, index) order."""
    out: list[dict] = []
    for iid in sorted(instances):
        for st in instances[iid].get("steps") or []:
            if st.get("executed") and st.get("exit_code") == 0 and st.get("kind") == "run":
                out.append({"binding": st["binding"], "writes": list(st["writes"]),
                            "deletes": list(st["deletes"]), "external": [],
                            "instance": iid, "index": st["index"]})
    return out


def inside_git_repo(path: Path) -> bool:
    p = path.resolve()
    for d in [p, *p.parents]:
        if (d / ".git").exists():
            return True
    return False


def merge_trace(existing: Mapping[str, Any] | None, benchmark: str, instance_id: str,
                entry: Mapping[str, Any]) -> dict:
    doc = dict(existing or {})
    doc["format"] = FORMAT
    doc["benchmark"] = benchmark
    instances = dict(doc.get("instances") or {})
    instances[instance_id] = dict(entry)
    doc["instances"] = {k: instances[k] for k in sorted(instances)}
    doc["steps"] = loader_steps(doc["instances"])
    return doc


def trace_path(line: ScopeLine, traces_dir: Path = TRACES_DIR) -> Path:
    return traces_dir / repo_dirname(line.repo) / (line.sha or "") / f"{line.benchmark_id()}.json"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _run(args: list[str], cwd: Path, env: Mapping[str, str] | None = None) -> None:
    subprocess.run(args, cwd=cwd, env=dict(env) if env is not None else None, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def clone(repo: str, sha: str, dest: Path) -> None:
    dest.mkdir(parents=True)
    _run(["git", "init", "-q"], dest)
    _run(["git", "remote", "add", "origin", f"https://github.com/{repo}.git"], dest)
    _run(["git", "fetch", "-q", "--depth", "1", "origin", sha], dest)
    _run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], dest)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True,
                          check=True).stdout.strip()
    if head != sha:
        raise SystemExit(f"error: clone is at {head}, expected {sha}")


def make_shim(shims: Path, pm: tuple[str, str] | None) -> str:
    shims.mkdir(parents=True, exist_ok=True)
    if pm is None:
        return "host (no packageManager field)"
    name, version = pm
    script = shims / name
    script.write_text(f'#!/bin/sh\nexec npx --yes {name}@{version} "$@"\n', encoding="utf-8")
    script.chmod(0o755)
    return f"{name}@{version} via npx"


def _env_value(v: str, unresolved: list[str]) -> str:
    for m in _EXPR.findall(v):
        unresolved.append(m)
    return _EXPR.sub("", v)


def trace_instance(line: ScopeLine, instance_spec: str, *, job_key: str | None, event: str,
                   workdir: Path | None, keep: bool, repos_dir: Path = REPOS_DIR) -> dict:
    from ghav import cost
    from ghav.lower import lower

    snap = snapshot_dir(line.repo, line.sha or "", repos_dir)
    program = lower(snap / (line.path or ""), snap)
    job = select_instance(program, instance_spec, job_key)
    run = cost.simulate(program, event)[job.id]
    if not run.ran:
        raise SystemExit(f"error: {job.id} does not run for event {event}")

    base = Path(tempfile.mkdtemp(prefix="ghav-trace-", dir=str(workdir) if workdir else None))
    if inside_git_repo(base):
        shutil.rmtree(base, ignore_errors=True)
        raise SystemExit(f"error: work directory {base} is inside a git repository")
    ws, home, tmp, shims, logs = (base / n for n in ("workspace", "home", "tmp", "shims", "logs"))
    for d in (home, tmp, logs):
        d.mkdir(parents=True)
    try:
        clone(line.repo, line.sha or "", ws)
        pm = package_manager(ws / "package.json")
        pm_desc = make_shim(shims, pm)
        node = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip() or None
        base_env = {
            "PATH": f"{shims}{os.pathsep}{os.environ.get('PATH', '')}",
            "HOME": str(home), "TMPDIR": str(tmp), "RUNNER_TEMP": str(tmp),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "CI": "true", "GITHUB_ACTIONS": "true", "GITHUB_WORKSPACE": str(ws),
            "GITHUB_EVENT_NAME": event, "GITHUB_SHA": line.sha or "", "GITHUB_REPOSITORY": line.repo,
            "npm_config_yes": "true",
        }
        snapper = Snapshotter(ws)
        current = snapper.snapshot()
        steps: list[dict] = []
        failed = False
        for i, step in enumerate(job.steps):
            b = step.binding
            entry: dict[str, Any] = {"index": i, "origin": list(step.origin), "name": b.name,
                                     "kind": b.kind, "binding": binding_json(b)}
            if b.kind != "run":
                entry.update(executed=False, skipped_reason="action", known_effects=known_effects(b.text))
                steps.append(entry)
                continue
            if not run.steps[i]:
                entry.update(executed=False, skipped_reason="condition")
                steps.append(entry)
                continue
            if failed:
                entry.update(executed=False, skipped_reason="previous-failure")
                steps.append(entry)
                continue
            unresolved: list[str] = []
            env = dict(base_env)
            for k, v in [*program.env, *job.env, *b.env]:
                env[str(k)] = _env_value(str(v), unresolved)
            script_text = _env_value(b.text, unresolved)
            if b.shell not in (None, "bash", "sh"):
                entry.update(executed=False, skipped_reason=f"unsupported shell {b.shell}")
                steps.append(entry)
                failed = True
                continue
            script = tmp / f"step{i}.sh"
            script.write_text(script_text, encoding="utf-8")
            shell = list(DEFAULT_SHELL) if b.shell in (None, "bash") else ["sh", "-e"]
            cwd = ws / (b.working_directory or ".")
            t0 = time.monotonic()
            with open(logs / f"step{i}.log", "wb") as log_f:
                proc = subprocess.run([*shell, str(script)], cwd=cwd, env=env, stdout=log_f,
                                      stderr=subprocess.STDOUT, check=False)
            elapsed = round(time.monotonic() - t0, 1)
            after = snapper.snapshot()
            entry.update(executed=True, exit_code=proc.returncode, wall_seconds=elapsed,
                         unresolved_expressions=sorted(set(unresolved)), **diff(current, after))
            current = after
            steps.append(entry)
            tail = (logs / f"step{i}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
            print(f"[trace] {job.id} step {i} ({b.name or b.text.splitlines()[0]}): exit {proc.returncode}, "
                  f"{elapsed}s, {entry['counts']}", file=sys.stderr)
            for t in tail:
                print(f"    | {t}", file=sys.stderr)
            if proc.returncode != 0:
                failed = True
        return {
            "instance": job.id,
            "check_name": job.check_name,
            "matrix": {k: v for k, v in job.matrix},
            "event": event,
            "complete": not failed,
            "environment": {"platform": f"{platform.system()}-{platform.machine()}", "node": node,
                            "package_manager": pm_desc, "runs_on_in_workflow": job.runs_on,
                            "head": line.sha},
            "steps": steps,
        }
    finally:
        if keep:
            print(f"[trace] kept work directory {base}", file=sys.stderr)
        else:
            shutil.rmtree(base, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--benchmark", required=True, help="scope line owner/repo@sha:.github/workflows/x.yml")
    ap.add_argument("--instance", required=True, help="matrix assignment k=v,... or an instance id")
    ap.add_argument("--job", default=None, help="original job key, when the assignment is ambiguous")
    ap.add_argument("--event", default="push", choices=("push", "pull_request"))
    ap.add_argument("--workdir", type=Path, default=None,
                    help="parent of the temporary directory (default: system temp); must be outside git repos")
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    ap.add_argument("--traces-dir", type=Path, default=TRACES_DIR)
    ap.add_argument("--repos-dir", type=Path, default=REPOS_DIR)
    args = ap.parse_args()

    line = ScopeLine.parse(args.benchmark)
    if line.sha is None or line.path is None:
        raise SystemExit("error: --benchmark must be a pinned workflow line")
    entry = trace_instance(line, args.instance, job_key=args.job, event=args.event,
                           workdir=args.workdir, keep=args.keep, repos_dir=args.repos_dir)
    out = trace_path(line, args.traces_dir)
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    doc = merge_trace(existing, str(line), entry["instance"], entry)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps_json(doc), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    return 0 if entry["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
