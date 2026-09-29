"""Contract inference from the global contract guide (docs/CONTRACT_GUIDE.md).

`infer(step, job, repo_root)` gives one step occurrence its region-effect contract;
`annotate(program, repo_root)` fills the contract of every step (frozen jobs included).
The procedure is repository independent and mechanical:

1. A step is analyzed as a list of commands: a `run` script is split into shell commands
   (guide 4.1), a local composite action is the union of its `action.yml` steps (guide 4.4),
   and any other `uses` reference is looked up in the action table.
2. Package-manager script invocations resolve through the root `package.json` (guide 4.2),
   including `pre`/`post` hooks, with depth at most `MAX_DEPTH`; cycles and missing scripts
   give top.
3. Each leaf command is matched against the declarative template table below (guide 5).
   The first matching row wins, so more specific rows come first. Unmatched commands give top.
4. Context reads follow guide 3: `ctx:platform` always, every `${{ }}` reference, job- and
   workflow-level env names appearing in the resolved text, and job-identity variables.
5. The contract is the union over all commands; any top command makes the step top.

The tables (`TEMPLATES`, `ACTION_ROWS`, `COMMAND_ROWS`, `RUNNER_ROWS`, `NOOP_PROGRAMS`)
are the single source of truth for the guide's section 5 and are dumped for auditing by
`template_table_markdown()`. Where the guide is silent, this module either applies the
guide's fallback (top) or adds reads/writes, never removes them; every such decision is
listed in `IMPLEMENTATION_NOTES` and printed with the table.

Determinism: no hash-order iteration reaches an output; all region sets are frozensets and
all lists that are reported are built in source order or sorted.
"""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from .wir import (
    CTX,
    DEP,
    EXT,
    OUT,
    OUTPUT,
    RUN,
    SRC,
    ST_JOB,
    ST_STEP,
    TOOL,
    TOP_CONTRACT,
    Binding,
    Contract,
    JobInstance,
    Program,
    Region,
    Step,
)

__all__ = [
    "MAX_DEPTH",
    "Template",
    "ActionRow",
    "CommandRow",
    "RunnerRow",
    "TEMPLATES",
    "ACTION_ROWS",
    "COMMAND_ROWS",
    "RUNNER_ROWS",
    "NOOP_PROGRAMS",
    "IMPLEMENTATION_NOTES",
    "ShellCommand",
    "CommandReport",
    "StepReport",
    "split_commands",
    "path_kinds",
    "FILE_UTILITY_ROWS",
    "FileUtilityRow",
    "ObservedStep",
    "TRACE_FORMAT",
    "TRANSFER_PREFIX",
    "TransferStep",
    "cache_violation",
    "explain_program",
    "load_observed",
    "parse_transfer",
    "trace_path",
    "transfer_bindings",
    "valid_transfer_name",
    "valid_transfer_path",
    "writes_dep_after_restore",
    "expression_refs",
    "explain",
    "infer",
    "annotate",
    "report",
    "template_table_markdown",
]

MAX_DEPTH = 5  # guide 4.2: script resolution depth; also bounds nested local actions

PLATFORM = Region(CTX, None, "platform")
JOB_IDENTITY = Region(CTX, None, "job-identity")

# ---------------------------------------------------------------------------
# Declarative tables (guide section 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    """One row of the guide's template table.

    Region specs: a job-local kind (`src`, `dep`, `out`, `tool`), `ctx:<key>`, `ext:<name>`,
    or one of the parametric specs `ext:{service}` (the matched row's services),
    `run:{with.name}` (artifact name) and `kinds:{with.path}` (kinds covering the path).
    `flag_writes` are extra writes when a flag is present on the command line.
    `out_paths`: "" (not transferable), "build" (explicit output flag, else tsconfig
    `outDir`, else `dist`), or "with.path".
    """

    name: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    out_paths: str = ""
    flag_writes: tuple[tuple[str, str], ...] = ()
    note: str = ""


TEMPLATES: tuple[Template, ...] = (
    Template("checkout", ("ctx:github.sha", "ctx:github.ref", "src", "dep", "out", "tool"),
             ("src", "dep", "out", "tool"),
             note="resets the workspace: reads and writes every workspace-resident kind; "
                  "reads `ext:refs` when `fetch-depth` is 0 (or an expression)"),
    Template("setup-runtime", ("src",), ("dep",), note="reads version files"),
    Template("install-pinned", ("src", "dep"), ("dep", "tool")),
    Template("install-unpinned", ("src", "dep", "ext:registry"), ("dep", "tool")),
    Template("build", ("src", "dep", "out"), ("out", "tool"), out_paths="build"),
    Template("build-orchestrated", ("src", "dep", "out"), ("out", "tool", "src"),
             note="not transferable"),
    Template("typecheck", ("src", "dep", "out"), ("tool",)),
    Template("lint", ("src", "dep", "out"), (),
             flag_writes=(("--cache", "tool"), ("--output-file", "tool"), ("-o", "tool"))),
    Template("format-write", ("src", "dep"), ("src", "tool")),
    Template("test", ("src", "dep", "out"), ("tool",),
             flag_writes=(("-u", "src"), ("--updateSnapshot", "src"),
                          ("--update-snapshots", "src"), ("--update", "src"))),
    Template("upload-artifact", ("kinds:{with.path}",), ("run:{with.name}",)),
    Template("download-artifact", ("run:{with.name}",), ("target:{with.path}",),
             out_paths="with.path",
             note="writes `out`, or `tool` when `with.path` is outside the workspace"),
    Template("cache", (), (), note="cold-cache semantics: no correctness effect"),
    Template("cache-opaque", ("src", "dep", "out", "tool"), ("dep", "tool", "kinds:{with.path}"),
             note="a cache restore followed by a later dep writer in its job (guide 7)"),
    Template("report-upload", ("src", "tool", "out", "ctx:github.sha"), ("ext:{service}",)),
    Template("publish", ("src", "dep", "out", "ctx:github.ref"), ("ext:{service}",)),
    Template("noop", (), ()),
    Template("file-util", ("paths:{sources}",), ("paths:{targets}",),
             note="guide 4.5: regions of the path arguments; unknown or variable paths give top"),
    Template("observed", ("src", "dep", "out", "tool"), ("paths:{trace}",),
             note="guide 8: write set from a filesystem trace, conservative reads"),
)

TEMPLATE_BY_NAME: Mapping[str, Template] = {t.name: t for t in TEMPLATES}


@dataclass(frozen=True)
class ActionRow:
    """An action reference pattern. `ref` is `owner/repo[/path]` (case-insensitive) or
    `owner/*`; any `@version` matches. `top_if` lists `(with-key, mode)` conditions that
    make the step top: mode `any` (key present), `literal` (present without `${{ }}`),
    `truthy` (present and not `false`), `missing` (key absent). `check` names an extra
    predicate on `with` that must hold (see `_ACTION_CHECKS`). `reads_if` lists
    `(with-key, mode, region)`: the region is also read when the key's value matches the mode
    (`zero`: the literal `0` or an expression)."""

    ref: str
    template: str
    services: tuple[str, ...] = ()
    top_if: tuple[tuple[str, str], ...] = ()
    check: str = ""
    reads_if: tuple[tuple[str, str, str], ...] = ()


ACTION_ROWS: tuple[ActionRow, ...] = (
    ActionRow("actions/checkout", "checkout",
              top_if=(("repository", "literal"), ("ref", "literal")),
              reads_if=(("fetch-depth", "zero", "ext:refs"),)),
    ActionRow("actions/setup-node", "setup-runtime"),
    ActionRow("actions/setup-python", "setup-runtime"),
    ActionRow("actions/setup-go", "setup-runtime"),
    ActionRow("actions/setup-java", "setup-runtime"),
    ActionRow("pnpm/action-setup", "setup-runtime", top_if=(("run_install", "truthy"),)),
    ActionRow("oven-sh/setup-bun", "setup-runtime"),
    ActionRow("denoland/setup-deno", "setup-runtime"),
    ActionRow("actions/upload-artifact", "upload-artifact", top_if=(("path", "missing"),)),
    ActionRow("actions/download-artifact", "download-artifact",
              top_if=(("name", "missing"), ("pattern", "any"), ("run-id", "any"))),
    ActionRow("actions/cache", "cache"),
    ActionRow("actions/cache/restore", "cache"),
    ActionRow("actions/cache/save", "cache"),
    ActionRow("codecov/codecov-action", "report-upload", services=("codecov",)),
    ActionRow("coverallsapp/github-action", "report-upload", services=("coveralls",)),
    ActionRow("SonarSource/*", "report-upload", services=("sonar",)),
    ActionRow("actions/upload-pages-artifact", "report-upload", services=("github-pages",)),
    ActionRow("actions/deploy-pages", "publish", services=("github-pages",)),
    ActionRow("peaceiris/actions-gh-pages", "publish", services=("github-pages",)),
    ActionRow("JamesIves/github-pages-deploy-action", "publish", services=("github-pages",)),
    ActionRow("softprops/action-gh-release", "publish", services=("github-releases",)),
    ActionRow("actions/github-script", "noop", check="script-only-logs"),
)


@dataclass(frozen=True)
class CommandRow:
    """A command pattern.

    `program`: executable name (after stripping `node_modules/.bin/`). `sub`: accepted
    leading positional words; an entry may hold several space-separated words, `""` means
    "no positional word at all", and `()` means "any". `any_flags`: at least one present.
    `no_flags`: none present (such commands fall through, usually to top). `positional`:
    constraint on positional words after `sub` (`any`, `none`, `some`). `no_positional`: none
    of the positional words may be one of these. `when`: a named
    repository condition (`yarn-berry`, `yarn-classic`). `lifecycle`: the command runs the
    root package's install lifecycle scripts unless `--ignore-scripts` is given.
    `extra_writes`: region specs added to the template's writes."""

    program: str
    template: str
    sub: tuple[str, ...] = ()
    any_flags: tuple[str, ...] = ()
    no_flags: tuple[str, ...] = ()
    positional: str = "any"
    no_positional: tuple[str, ...] = ()
    when: str = ""
    services: tuple[str, ...] = ()
    extra_writes: tuple[str, ...] = ()
    lifecycle: bool = False
    tsconfig: bool = False
    out_flags: tuple[str, ...] = ()
    note: str = ""


_NPM_INSTALL = ("install", "i", "add", "in")
_PNPM_NON_SCRIPT = (
    "add", "dlx", "exec", "i", "import", "install", "link", "pack", "patch", "publish",
    "rebuild", "remove", "rm", "uninstall", "unlink", "up", "update", "upgrade")
_WATCH = ("--watch", "-w", "--watchAll")
_LINT_WRITE_FLAGS = ("--fix", "--write", "--apply", "--apply-unsafe", "--fix-dangerously",
                     "--fix-suggestions", "--unsafe")

COMMAND_ROWS: tuple[CommandRow, ...] = (
    # install-pinned
    CommandRow("npm", "install-pinned", sub=("ci", "clean-install", "ic", "install-clean"),
               lifecycle=True),
    CommandRow("pnpm", "install-pinned", sub=("install", "i"), no_flags=("--no-frozen-lockfile",),
               positional="none", lifecycle=True, note="CI default is --frozen-lockfile"),
    CommandRow("yarn", "install-pinned", sub=("", "install"),
               any_flags=("--frozen-lockfile", "--immutable"), positional="none", lifecycle=True),
    CommandRow("yarn", "install-pinned", sub=("", "install"), positional="none",
               when="yarn-berry", lifecycle=True, note="berry is immutable in CI"),
    CommandRow("bun", "install-pinned", sub=("install", "i"), any_flags=("--frozen-lockfile",),
               positional="none", lifecycle=True),
    CommandRow("nci", "install-pinned", lifecycle=True),
    # install-unpinned
    CommandRow("npm", "install-unpinned", sub=_NPM_INSTALL, any_flags=("-g", "--global")),
    CommandRow("npm", "install-unpinned", sub=_NPM_INSTALL, any_flags=("--no-save",)),
    CommandRow("npm", "install-unpinned", sub=_NPM_INSTALL, positional="some",
               extra_writes=("src",), note="adds packages to package.json"),
    CommandRow("npm", "install-unpinned", sub=_NPM_INSTALL, positional="none", lifecycle=True),
    CommandRow("pnpm", "install-unpinned", sub=("add",), any_flags=("-g", "--global")),
    CommandRow("pnpm", "install-unpinned", sub=("add",), extra_writes=("src",)),
    CommandRow("pnpm", "install-unpinned", sub=("install", "i"), positional="none",
               lifecycle=True, note="--no-frozen-lockfile"),
    CommandRow("yarn", "install-unpinned", sub=("global add",)),
    CommandRow("yarn", "install-unpinned", sub=("add",), extra_writes=("src",)),
    CommandRow("yarn", "install-unpinned", sub=("", "install"), positional="none",
               when="yarn-classic", lifecycle=True, note="classic is not frozen by default"),
    CommandRow("bun", "install-unpinned", sub=("install", "i"), positional="none",
               lifecycle=True),
    CommandRow("bun", "install-unpinned", sub=("add",), extra_writes=("src",)),
    # build-orchestrated (before plain pnpm rows are dispatched as scripts)
    CommandRow("turbo", "build-orchestrated", sub=("run",)),
    CommandRow("nx", "build-orchestrated", sub=("run", "run-many", "affected")),
    CommandRow("lerna", "build-orchestrated", sub=("run",)),
    CommandRow("pnpm", "publish", sub=("publish",), any_flags=("-r", "--recursive"),
               services=("npm-registry",)),
    CommandRow("pnpm", "build-orchestrated", any_flags=("-r", "--recursive", "--filter", "-F"),
               no_positional=_PNPM_NON_SCRIPT, note="recursive/filtered script runs"),
    # typecheck before build: same programs
    CommandRow("tsc", "typecheck", any_flags=("--noEmit", "--noemit"), no_flags=_WATCH + ("--init",)),
    CommandRow("vue-tsc", "typecheck", any_flags=("--noEmit", "--noemit"), no_flags=_WATCH),
    # build
    CommandRow("tsc", "build", no_flags=_WATCH + ("--init",), tsconfig=True,
               out_flags=("--outDir", "--outdir"),
               note="typecheck instead when the tsconfig sets noEmit"),
    CommandRow("vite", "build", sub=("build",), no_flags=_WATCH, out_flags=("--outDir",)),
    CommandRow("rollup", "build", no_flags=_WATCH, out_flags=("-d", "--dir", "-o", "--file")),
    CommandRow("tsup", "build", no_flags=_WATCH, out_flags=("-d", "--out-dir")),
    CommandRow("esbuild", "build", no_flags=("--watch", "--serve"),
               out_flags=("--outdir", "--outfile")),
    CommandRow("webpack", "build", no_flags=_WATCH, out_flags=("-o", "--output-path")),
    CommandRow("unbuild", "build"),
    CommandRow("babel", "build", any_flags=("-d", "--out-dir"), no_flags=_WATCH,
               out_flags=("-d", "--out-dir")),
    # format-write before lint: same programs
    CommandRow("prettier", "format-write", any_flags=("--write", "-w")),
    CommandRow("biome", "format-write", sub=("format",), any_flags=("--write",)),
    CommandRow("eslint", "format-write", any_flags=("--fix",)),
    # lint (write flags not listed in the guide fall through to top)
    CommandRow("eslint", "lint", no_flags=_LINT_WRITE_FLAGS),
    CommandRow("prettier", "lint", any_flags=("--check", "-c", "--list-different", "-l")),
    CommandRow("biome", "lint", sub=("check", "lint", "ci"), no_flags=_LINT_WRITE_FLAGS),
    CommandRow("oxlint", "lint", no_flags=_LINT_WRITE_FLAGS),
    CommandRow("stylelint", "lint", no_flags=_LINT_WRITE_FLAGS),
    CommandRow("xo", "lint", no_flags=_LINT_WRITE_FLAGS),
    CommandRow("oxfmt", "lint", any_flags=("--check",)),
    CommandRow("madge", "lint", no_flags=("--image", "--dot")),
    CommandRow("publint", "lint", no_flags=_LINT_WRITE_FLAGS),
    CommandRow("knip", "lint", no_flags=_LINT_WRITE_FLAGS),
    # test
    CommandRow("jest", "test", no_flags=("--watch", "--watchAll")),
    CommandRow("vitest", "test", no_flags=("--watch", "-w")),
    CommandRow("mocha", "test", no_flags=("--watch", "-w")),
    CommandRow("ava", "test", no_flags=("--watch", "-w")),
    CommandRow("node", "test", any_flags=("--test",), no_flags=("--watch",)),
    CommandRow("playwright", "test", sub=("test",)),
    CommandRow("cypress", "test", sub=("run",)),
    # publish
    CommandRow("npm", "publish", sub=("publish",), services=("npm-registry",)),
    CommandRow("pnpm", "publish", sub=("publish",), services=("npm-registry",)),
    CommandRow("yarn", "publish", sub=("publish", "npm publish"), services=("npm-registry",)),
    CommandRow("changeset", "publish", sub=("publish",), services=("npm-registry",)),
    CommandRow("gh", "publish", sub=("release",), services=("github-releases",)),
    CommandRow("semantic-release", "publish", services=("github-releases", "npm-registry")),
)


@dataclass(frozen=True)
class RunnerRow:
    """A package runner prefix (`npx tsc ...`). The wrapped command is matched on its own.
    mode `npx`: runs the local binary when its package is a root dependency, otherwise
    downloads it (adds install-unpinned); `local`: never downloads; `download`: always
    downloads. `value_flags` take a value; `package_flags` name the package explicitly."""

    prefix: tuple[str, ...]
    mode: str
    value_flags: tuple[str, ...] = ()
    package_flags: tuple[str, ...] = ()
    top_flags: tuple[str, ...] = ()


RUNNER_ROWS: tuple[RunnerRow, ...] = (
    RunnerRow(("npx",), "npx", value_flags=("-p", "--package"), package_flags=("-p", "--package"),
              top_flags=("-c", "--call")),
    RunnerRow(("npm", "exec"), "npx", value_flags=("-p", "--package"),
              package_flags=("-p", "--package"), top_flags=("-c", "--call", "-w", "--workspace",
                                                             "--workspaces", "-ws")),
    RunnerRow(("npm", "x"), "npx", value_flags=("-p", "--package"),
              package_flags=("-p", "--package"), top_flags=("-c", "--call")),
    RunnerRow(("pnpm", "exec"), "local", top_flags=("-r", "--recursive", "--filter", "-F", "-C",
                                                     "--dir", "-c", "--shell-mode")),
    RunnerRow(("pnpm", "dlx"), "download", value_flags=("--package",),
              package_flags=("--package",), top_flags=("-c", "--shell-mode")),
    RunnerRow(("pnpx",), "download"),
    RunnerRow(("yarn", "exec"), "local"),
    RunnerRow(("yarn", "dlx"), "download", value_flags=("-p", "--package"),
              package_flags=("-p", "--package")),
    RunnerRow(("bunx",), "npx", value_flags=("-p", "--package"), package_flags=("-p", "--package")),
    RunnerRow(("bun", "x"), "npx", value_flags=("-p", "--package"),
              package_flags=("-p", "--package")),
    RunnerRow(("nlx",), "download"),
)

# Guide 4.1: shell-only commands. `cd` additionally changes the working directory; `exit` and
# `false` only set the step's status, which the semantics models by the task's `ok`.
NOOP_PROGRAMS: tuple[str, ...] = (
    ":", "cd", "echo", "exit", "export", "false", "printf", "pwd", "set", "shopt", "true", "unset",
    "Write-Host", "Write-Output",
)

# Workflow commands printed by echo/printf that change state (and therefore are not noops).
_STATEFUL_WORKFLOW_COMMANDS = ("::set-output", "::set-env", "::add-path", "::save-state")


@dataclass(frozen=True)
class FileUtilityRow:
    """A file utility (guide 4.5). `sub` is a required first word (`shx rm`). `role` says which
    positional arguments are sources (read) and which are targets (written):
    `targets` (every path is written), `copy` (sources..., target; or `-t DIR`), `move` (like
    copy, sources are also written), `link` (target, link name), `sed` (`-i` required; files
    are read and written), `json` (`-I -f FILE` required; FILE is read and written), `tar`
    (`-c`: members read, archive written; `-x` with explicit members: archive read, members
    written). `value_flags` take a separate value; `read_flags` name a file that is read."""

    program: str
    role: str
    sub: str = ""
    value_flags: tuple[str, ...] = ()
    read_flags: tuple[str, ...] = ()
    top_flags: tuple[str, ...] = ()
    note: str = ""


FILE_UTILITY_ROWS: tuple[FileUtilityRow, ...] = (
    FileUtilityRow("rm", "targets"),
    FileUtilityRow("rimraf", "targets", top_flags=("--preserve-root=false",)),
    FileUtilityRow("del", "targets", top_flags=("--cwd",)),
    FileUtilityRow("del-cli", "targets", top_flags=("--cwd",)),
    FileUtilityRow("shx", "targets", sub="rm"),
    FileUtilityRow("shx", "copy", sub="cp"),
    FileUtilityRow("shx", "move", sub="mv"),
    FileUtilityRow("shx", "targets", sub="mkdir"),
    FileUtilityRow("shx", "targets", sub="touch"),
    FileUtilityRow("cp", "copy", value_flags=("-t", "--target-directory", "-S", "--suffix")),
    FileUtilityRow("mv", "move", value_flags=("-t", "--target-directory", "-S", "--suffix")),
    FileUtilityRow("mkdir", "targets", value_flags=("-m", "--mode")),
    FileUtilityRow("touch", "targets", value_flags=("-d", "--date", "-t"),
                   read_flags=("-r", "--reference")),
    FileUtilityRow("ln", "link", value_flags=("-t", "--target-directory", "-S", "--suffix")),
    FileUtilityRow("sed", "sed", value_flags=("-e", "--expression", "-l", "--line-length"),
                   read_flags=("-f", "--file")),
    FileUtilityRow("json", "json", value_flags=("-e", "-c", "-d", "-D", "-a", "-k", "-n", "-o"),
                   read_flags=("-f",)),
    FileUtilityRow("tar", "tar"),
)

# Glob characters: a glob maps to the regions of its fixed prefix (guide 4.5).
_GLOB_CHARS = frozenset("*?[]{}")

# ---------------------------------------------------------------------------
# The transfer idiom (RESEARCH_SCOPE: one idiom; emitted by ghav.emit.transfer_steps)
# ---------------------------------------------------------------------------

TRANSFER_PREFIX = "ghav-"                 # reserved artifact-name prefix
TRANSFER_UPLOAD = "actions/upload-artifact@v4"
TRANSFER_DOWNLOAD = "actions/download-artifact@v4"
TRANSFER_TEMP = "${{ runner.temp }}"
TRANSFER_SHELL = "bash"
_TRANSFER_NAME = re.compile(r"ghav-[A-Za-z0-9._-]{1,120}\Z")
_TRANSFER_TAR = re.compile(
    r'tar -(?P<mode>[cx])f "\$RUNNER_TEMP/(?P<name>ghav-[A-Za-z0-9._-]+)\.tar" '
    r'-C "\$GITHUB_WORKSPACE" (?P<paths>.+)\Z', re.S)


@dataclass(frozen=True)
class TransferStep:
    """One step of the transfer idiom: `pack` (tar the declared out paths relative to the
    workspace root into `$RUNNER_TEMP/<name>.tar`), `upload` (that file, with
    `if-no-files-found: error` and `retention-days: 1`), `download` (into `$RUNNER_TEMP`) and
    `unpack` (extract exactly the declared paths into the workspace root)."""

    role: str
    name: str
    paths: tuple[str, ...] = ()


def valid_transfer_name(name: str) -> bool:
    """A transfer artifact name: the reserved prefix `ghav-` plus `[A-Za-z0-9._-]`."""
    return bool(_TRANSFER_NAME.match(name or ""))


def valid_transfer_path(path: str) -> bool:
    """A declared out path the idiom can carry: relative, normalized, inside the workspace,
    no glob, variable, quote or whitespace, and not starting with `-`."""
    if not path or path != posixpath.normpath(path) or path in (".", "..") \
            or path.startswith(("/", "../", "-", "~")):
        return False
    return not any(c in path for c in "*?[]{}$`'\"\\ \t\n!")


def transfer_bindings(name: str, paths: Sequence[str]) -> tuple[Binding, Binding, Binding, Binding]:
    """The four bindings of the idiom (pack, upload, download, unpack) for `name`, carrying the
    workspace-relative `paths`. Raises ValueError on an invalid name or path."""
    if not valid_transfer_name(name):
        raise ValueError(f"invalid transfer artifact name {name!r}")
    ps = tuple(sorted(set(paths)))
    if not ps or not all(valid_transfer_path(x) for x in ps):
        raise ValueError(f"invalid transfer paths {list(paths)!r}")
    if any(a != b and b.startswith(a + "/") for a in ps for b in ps):
        raise ValueError(f"nested transfer paths {list(ps)!r}")
    members = " ".join(ps)
    archive = f'"$RUNNER_TEMP/{name}.tar"'
    pack = Binding("run", f'tar -cf {archive} -C "$GITHUB_WORKSPACE" {members}',
                   shell=TRANSFER_SHELL, name=f"ghav: pack {name}")
    upload = Binding("uses", TRANSFER_UPLOAD, tuple(sorted((
        ("name", name), ("path", f"{TRANSFER_TEMP}/{name}.tar"), ("if-no-files-found", "error"),
        ("retention-days", "1")))), name=f"ghav: upload {name}")
    download = Binding("uses", TRANSFER_DOWNLOAD, (("name", name), ("path", TRANSFER_TEMP)),
                       name=f"ghav: download {name}")
    unpack = Binding("run", f'tar -xf {archive} -C "$GITHUB_WORKSPACE" {members}',
                     shell=TRANSFER_SHELL, name=f"ghav: unpack {name}")
    return pack, upload, download, unpack


def parse_transfer(b: Binding) -> TransferStep | None:
    """The idiom step `b` is (compared on `Binding.key()`, so names do not matter), or None."""
    if b.kind == "run":
        m = _TRANSFER_TAR.match(b.text or "")
        if m is None:
            return None
        paths = tuple(m.group("paths").split(" "))
        try:
            bindings = transfer_bindings(m.group("name"), paths)
        except ValueError:
            return None
        role, ref = ("pack", bindings[0]) if m.group("mode") == "c" else ("unpack", bindings[3])
        return TransferStep(role, m.group("name"), paths) if ref.key() == b.key() else None
    if b.kind == "uses" and b.text in (TRANSFER_UPLOAD, TRANSFER_DOWNLOAD):
        name = dict(b.with_).get("name", "")
        if not valid_transfer_name(name):
            return None
        _, upload, download, _ = transfer_bindings(name, ("x",))
        if b.key() == upload.key():
            return TransferStep("upload", name)
        if b.key() == download.key():
            return TransferStep("download", name)
    return None

# Script dispatch (guide 4.2).
_NPM_RUN = ("run", "run-script", "rum", "urn")
_NPM_LIFECYCLE = {"test": "test", "t": "test", "tst": "test", "start": "start", "stop": "stop"}
_PNPM_BUILTINS = frozenset(
    "add audit bin cat-file cat-index config create dedupe deploy dlx doctor env exec fetch "
    "find-hash get i import init install install-test it licenses link list ln ls outdated "
    "pack patch patch-commit patch-remove prune publish rb rebuild recursive remove rm root "
    "sbom server set setup store un uninstall unlink up update upgrade why self-update "
    "approve-builds ignored-builds catalog".split())
_YARN_BUILTINS = frozenset(
    "add audit autoclean bin cache check config constraints create dedupe dlx exec explain "
    "generate-lock-entry global help import info init install licenses link list login logout "
    "node npm outdated owner pack patch patch-commit plugin policies publish rebuild remove "
    "sdks search set stage tag team unlink unplug up upgrade upgrade-interactive version "
    "versions why workspace workspaces".split())
_PM_SCRIPT_ALIASES = {"test": "test", "t": "test", "tst": "test", "start": "start"}
_PM_DIR_FLAGS = ("-C", "--dir", "--prefix", "--cwd", "-w", "--workspace", "--workspaces", "-ws",
                 "--filter", "-F", "-r", "--recursive", "--workspace-dir")
# Root lifecycle scripts run by an install without `--ignore-scripts`.
_INSTALL_LIFECYCLE = ("preinstall", "install", "postinstall", "prepublish", "preprepare",
                      "prepare", "postprepare")

# Binary name -> packages that provide it, when they differ (used by the npx rule only).
_BIN_PACKAGES: Mapping[str, tuple[str, ...]] = {
    "biome": ("@biomejs/biome",),
    "changeset": ("@changesets/cli",),
    "playwright": ("@playwright/test", "playwright"),
    "tsc": ("typescript",),
}
_PACKAGE_BINS: Mapping[str, str] = {p: b for b, ps in _BIN_PACKAGES.items() for p in ps}

# Path-to-region mapping (guide 2).
_WORKSPACE_PREFIXES = ("${{ github.workspace }}", "${{github.workspace}}", "$GITHUB_WORKSPACE",
                       "${GITHUB_WORKSPACE}")
_OUTSIDE_PREFIXES = ("${{ runner.temp }}", "${{runner.temp}}", "$RUNNER_TEMP", "${RUNNER_TEMP}",
                     "~", "$HOME", "${HOME}")
_OUT_DIR_NAMES = ("build", "dist", "lib", "out")
_TOOL_NAMES = (".cache", ".eslintcache", ".nx", ".nyc_output", ".tmp", ".turbo", "blob-report",
               "coverage", "junit.xml", "playwright-report", "reports", "temp", "test-results",
               "tmp")
_TOOL_SUFFIXES = (".tsbuildinfo", ".lcov", "lcov.info")

# Context-read rules (guide 3).
_IDENTITY_ENV = frozenset({"GITHUB_JOB", "GITHUB_RUN_ID", "RUNNER_NAME", "GITHUB_ACTION"})
_IDENTITY_GITHUB_KEYS = frozenset({"job", "run_id", "action"})
_FILE_COMMAND_ENV = frozenset({"GITHUB_ENV", "GITHUB_OUTPUT", "GITHUB_PATH", "GITHUB_STATE",
                               "GITHUB_STEP_SUMMARY"})
_CONTEXT_NAMES = frozenset({"github", "env", "vars", "secrets", "matrix", "inputs", "strategy",
                            "runner", "job", "jobs", "steps", "needs"})

IMPLEMENTATION_NOTES: tuple[str, ...] = (
    "Commands are split on newlines, `;`, `&&`, `||` (guide 4.1) and also on `|`: every "
    "pipeline stage runs, so the step contract is the union of the stages.",
    "Shell constructs the guide does not model give top: background `&`, subshells, command "
    "substitution, here-documents, input redirection, and output redirection to anything but "
    "`/dev/null` or a file descriptor (this includes `>> $GITHUB_OUTPUT`).",
    "`echo`/`printf` printing `::set-output`, `::set-env`, `::add-path` or `::save-state` give top.",
    "Package runners (`npx`, `pnpm exec`, `pnpm dlx`, `yarn dlx`, `bunx`, ...) are transparent: "
    "the wrapped command is matched on its own; a download (package not a root dependency, or "
    "an always-downloading runner) adds install-unpinned.",
    "Script arguments are appended to the script body before matching (`npm run test -- -u`); "
    "`--if-present` turns a missing script into a noop.",
    "Install commands without `--ignore-scripts` (and without package arguments) also run the "
    "root lifecycle scripts preinstall, install, postinstall, prepublish, preprepare, prepare "
    "and postprepare, resolved like `npm run`.",
    "Installs that add packages to package.json (`npm install <pkg>` without --no-save/-g, "
    "`pnpm add`, `yarn add`, `bun add`) also write `src` (guide 2: tracked files).",
    "Pre/post hooks are included for every package manager (an over-approximation for managers "
    "that skip them).",
    "Script resolution and lifecycle scripts need the working directory to be the repository "
    "root (v1); `cd`, `working-directory` and package-manager directory flags otherwise give top.",
    "Build `out_paths`: explicit output flag, else tsconfig `outDir` (the `-p`/`--project` or "
    "`-b` config for tsc, `tsconfig.json` in the working directory otherwise; relative "
    "`extends` are followed), else `dist`, all relative to the working directory. A tsconfig "
    "with `noEmit: true` makes `tsc` a typecheck. Unknown or escaping directories make the "
    "output non-transferable.",
    "Context reads: `${{ }}` references in run/with/env/working-directory; `matrix.<k>` and "
    "`inputs.<k>` are keyed by their first segment, `github.*` by the full path, bare "
    "`matrix` reads every matrix key; `runner.*` reads `ctx:platform`; `strategy.*`, "
    "`github.job|run_id|action` and `runner.name` read `ctx:job-identity`; `hashFiles(...)` "
    "reads `src` and `out`; `steps.<id>.*` reads the status cell of that earlier step; "
    "`needs.<job>.outputs.<o>` reads the job output region and `needs.<job>.result` the "
    "job status cell when `<job>` names exactly one needed instance; other references "
    "(`job.*`, bare `github`, ...) give top.",
    "Default environment variables in resolved text mirror their contexts: `GITHUB_<X>` reads "
    "`ctx:github.<x>` (GITHUB_EVENT_PATH reads `ctx:github.event`), `RUNNER_*` reads "
    "`ctx:platform`; file-command variables (GITHUB_OUTPUT, GITHUB_ENV, ...) give top.",
    "Inside a local composite action, `inputs.*` are substituted from the caller's `with` "
    "(or the input default) and `steps.*` refer to the action's own steps (no read); the "
    "action steps' `if:` conditions contribute context reads.",
    "`${{ matrix.<k> }}` is substituted with the instance's matrix value before a run script "
    "is split, so matrix-selected commands are resolved per instance; other expressions are "
    "assumed to expand to a single word.",
    "Artifact names are resolved with the instance's matrix values; a name that still depends "
    "on the instance gives top.",
    "Flag-conditional writes beyond the guide: lint `-o`/`--output-file` writes `tool` (report "
    "files, guide 2); test snapshot-update flags (`-u`, `--update`, ...) write `src`.",
    "Cache actions contribute no context reads from `with` (their inputs only name the cache; "
    "cold-cache semantics), so e.g. `hashFiles` in a cache key is not a read.",
    "Report-upload and publish services: codecov, coveralls, sonar, github-pages, "
    "github-releases, npm-registry.",
    "A command that writes `out` without declaring its paths (e.g. build-orchestrated, or a "
    "download whose path is not a literal directory) leaves the whole step without "
    "`out_paths`: the content of `out` is then unknown to exact-copy transfers.",
    "File utilities (guide 4.5): the flags that take a separate value are listed per utility "
    "(`FILE_UTILITY_ROWS`); any other flag is boolean, so a misread value only adds a path "
    "(an over-approximation). `-C` directories of `tar` apply in order.",
    "Cache-restoring setup actions (guide 7) keep their `setup-runtime` template; the "
    "restore is recorded in the step report (`restores_cache`) and, when the rule fails, the "
    "template gets a `cache-opaque` component.",
)

# ---------------------------------------------------------------------------
# Shell splitting (guide 4.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellCommand:
    """One simple command. `words` have quotes removed; `problem` names an unsupported shell
    construct (the command is then top)."""

    words: tuple[str, ...]
    text: str
    problem: str | None = None


_WORD_BREAK = set(" \t\r\n;&|<>()")


def _read_word(s: str, i: int) -> tuple[str, int, str | None]:
    buf: list[str] = []
    problem: str | None = None
    n = len(s)
    while i < n:
        c = s[i]
        if c in _WORD_BREAK:
            break
        if c == "\\":
            if i + 1 < n:
                if s[i + 1] != "\n":
                    buf.append(s[i + 1])
                i += 2
            else:
                i += 1
            continue
        if c == "'":
            j = s.find("'", i + 1)
            if j < 0:
                problem = problem or "unterminated quote"
                buf.append(s[i + 1:])
                i = n
                break
            buf.append(s[i + 1:j])
            i = j + 1
            continue
        if c == '"':
            i += 1
            while i < n and s[i] != '"':
                ch = s[i]
                if ch == "\\" and i + 1 < n and s[i + 1] in '$`"\\\n':
                    if s[i + 1] != "\n":
                        buf.append(s[i + 1])
                    i += 2
                    continue
                if s.startswith("${{", i):
                    j = s.find("}}", i + 3)
                    j = n if j < 0 else j + 2
                    buf.append(s[i:j])
                    i = j
                    continue
                if ch == "`" or s.startswith("$(", i):
                    problem = problem or "command substitution"
                buf.append(ch)
                i += 1
            if i >= n:
                problem = problem or "unterminated quote"
            i += 1
            continue
        if s.startswith("${{", i):
            j = s.find("}}", i + 3)
            j = n if j < 0 else j + 2
            buf.append(s[i:j])
            i = j
            continue
        if c == "`" or s.startswith("$(", i):
            problem = problem or "command substitution"
            buf.append(c)
            i += 1
            continue
        if s.startswith("${", i):
            j = s.find("}", i + 2)
            j = n if j < 0 else j + 1
            buf.append(s[i:j])
            i = j
            continue
        buf.append(c)
        i += 1
    return "".join(buf), i, problem


def split_commands(script: str) -> tuple[ShellCommand, ...]:
    """Split a shell script into simple commands (guide 4.1 plus pipelines)."""
    s = script
    n = len(s)
    out: list[ShellCommand] = []
    words: list[str] = []
    problem: str | None = None
    start = 0
    last_word_end = -1
    i = 0

    def flush(end: int) -> None:
        nonlocal words, problem
        text = s[start:end].strip()
        if words or problem:
            out.append(ShellCommand(tuple(words), text, problem))
        words = []
        problem = None

    def note(p: str) -> None:
        nonlocal problem
        problem = problem or p

    while i < n:
        c = s[i]
        if c in " \t\r":
            i += 1
            continue
        if s.startswith("\\\n", i):
            i += 2
            continue
        if c == "#":
            j = s.find("\n", i)
            i = n if j < 0 else j
            continue
        if c in "\n;":
            flush(i)
            i += 1
            start = i
            continue
        if s.startswith("&&", i) or s.startswith("||", i) or s.startswith("|&", i):
            flush(i)
            i += 2
            start = i
            continue
        if c == "|":
            flush(i)
            i += 1
            start = i
            continue
        if s.startswith("&>", i):
            j = i + 2
            if s.startswith(">", j):
                j += 1
            while j < n and s[j] in " \t":
                j += 1
            target, j, p = _read_word(s, j)
            if p:
                note(p)
            if target != "/dev/null":
                note(f"output redirection to {target or '?'}")
            i = j
            continue
        if c == "&":
            note("background process (&)")
            flush(i)
            i += 1
            start = i
            continue
        if c in "<>":
            if words and last_word_end == i and words[-1].isdigit():
                words.pop()  # file-descriptor prefix such as 2>
            j = i + 1
            dup = False
            if c == "<":
                if s.startswith("<<", i):
                    note("here-document")
                elif s.startswith("<(", i):
                    note("process substitution")
                else:
                    note("input redirection")
                while j < n and s[j] in "<&-(":
                    j += 1
            else:
                if j < n and s[j] in ">|":
                    j += 1
                if j < n and s[j] == "&":
                    dup = True
                    j += 1
                if j < n and s[j] == "(":
                    note("process substitution")
            while j < n and s[j] in " \t":
                j += 1
            target, j, p = _read_word(s, j)
            if p:
                note(p)
            if c == ">" and not (dup and (target.isdigit() or target == "-")) \
                    and target != "/dev/null":
                note(f"output redirection to {target or '?'}")
            i = j
            continue
        if c in "()":
            note("subshell or grouping")
            i += 1
            continue
        word, j, p = _read_word(s, i)
        if p:
            note(p)
        words.append(word)
        last_word_end = j
        i = j
    flush(n)
    return tuple(out)


# ---------------------------------------------------------------------------
# Expressions (guide 3.2)
# ---------------------------------------------------------------------------

_EXPR_SPAN = re.compile(r"\$\{\{(.*?)\}\}", re.S)
_EXPR_TOKEN = re.compile(
    r"\s*(?:(?P<str>'(?:[^']|'')*')|(?P<id>[A-Za-z_][A-Za-z0-9_-]*)|(?P<num>\d+(?:\.\d+)?)"
    r"|(?P<op>\*|\.|\[|\]|\(|\)|[^\sA-Za-z0-9_'.\[\]()]+))")


def _expr_tokens(text: str) -> list[tuple[str, str]]:
    toks: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _EXPR_TOKEN.match(text, pos)
        if not m or m.end() == pos:
            pos += 1
            continue
        pos = m.end()
        for kind in ("str", "id", "num", "op"):
            if m.group(kind) is not None:
                toks.append((kind, m.group(kind)))
                break
    return toks


def expression_refs(expr: str) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Context references and called function names of one expression body.

    Returns (refs, functions); each ref is its segment path, e.g. ("matrix", "node-version").
    `matrix['x']` gives ("matrix", "x"); `a[*].b` gives ("a", "*", "b"); a dynamic index ends
    the path.
    """
    toks = _expr_tokens(expr)
    refs: list[tuple[str, ...]] = []
    funcs: list[str] = []
    k = 0
    while k < len(toks):
        kind, val = toks[k]
        prev = toks[k - 1][1] if k > 0 else ""
        if kind != "id" or prev == ".":
            k += 1
            continue
        if k + 1 < len(toks) and toks[k + 1][1] == "(":
            funcs.append(val)
            k += 1
            continue
        if val not in _CONTEXT_NAMES:
            k += 1
            continue
        path = [val]
        k += 1
        while k < len(toks):
            if toks[k][1] == "." and k + 1 < len(toks) and toks[k + 1][0] in ("id", "num", "op"):
                nxt = toks[k + 1][1]
                if toks[k + 1][0] == "op" and nxt != "*":
                    break
                path.append(nxt)
                k += 2
                continue
            if toks[k][1] == "[" and k + 2 < len(toks) and toks[k + 2][1] == "]":
                kind2, v2 = toks[k + 1]
                if kind2 == "str":
                    path.append(v2[1:-1].replace("''", "'"))
                elif kind2 == "num" or v2 == "*":
                    path.append(v2)
                else:
                    break
                k += 3
                continue
            break
        refs.append(tuple(path))
    return tuple(refs), tuple(funcs)


def _expr_bodies(text: str) -> list[str]:
    return [m.group(1) for m in _EXPR_SPAN.finditer(text)]


_MATRIX_SIMPLE = re.compile(
    r"\$\{\{\s*matrix(?:\.([A-Za-z0-9_-]+)|\[\s*'([^']*)'\s*\])\s*\}\}")
_INPUTS_SIMPLE = re.compile(
    r"\$\{\{\s*inputs(?:\.([A-Za-z0-9_-]+)|\[\s*'([^']*)'\s*\])\s*\}\}")


def _substitute(pattern: re.Pattern[str], text: str, values: Mapping[str, str]) -> str:
    def rep(m: re.Match[str]) -> str:
        key = m.group(1) if m.group(1) is not None else m.group(2)
        return values[key] if key in values else m.group(0)
    return pattern.sub(rep, text)


# ---------------------------------------------------------------------------
# Path-to-region mapping (guide 2)
# ---------------------------------------------------------------------------

_ALL_LOCAL = frozenset({SRC, DEP, OUT, TOOL})


def path_kinds(path: str, out_dirs: Iterable[str] = ()) -> frozenset[str]:
    """Job-local kinds a workspace path may cover (guide 2). Unknown paths over-approximate."""
    p = path.strip()
    for prefix in ("${{ github.workspace }}/", "${{github.workspace}}/", "$GITHUB_WORKSPACE/",
                   "${GITHUB_WORKSPACE}/"):
        if p.startswith(prefix):
            p = p[len(prefix):]
    for prefix in ("${{ runner.temp }}", "${{runner.temp}}", "$RUNNER_TEMP", "${RUNNER_TEMP}",
                   "~", "$HOME", "${HOME}", "/"):
        if p.startswith(prefix):
            return frozenset({TOOL})
    if "$" in p or not p:
        return _ALL_LOCAL
    p = posixpath.normpath(p)
    if p.startswith(".."):
        return _ALL_LOCAL
    parts = [x for x in p.split("/") if x not in ("", ".")]
    if not parts or all(x in ("*", "**") for x in parts):
        return _ALL_LOCAL
    if "node_modules" in parts:
        return frozenset({DEP})
    outs = set(_OUT_DIR_NAMES) | set(out_dirs)
    if parts[0] in outs or p in outs:
        return frozenset({OUT})
    if any(x in _TOOL_NAMES or x.endswith(_TOOL_SUFFIXES) for x in parts):
        return frozenset({TOOL})
    if any(x in outs for x in parts[1:]) or any(
            posixpath.join(*parts[:k]) in outs for k in range(2, len(parts) + 1)):
        return frozenset({SRC, OUT})
    return frozenset({SRC, OUT, TOOL})


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandReport:
    """One analyzed command: where it came from, its text, and the outcome (a template name,
    `top`, or `resolve` for a script invocation that was expanded)."""

    origin: str
    text: str
    outcome: str
    detail: str = ""


@dataclass(frozen=True)
class StepReport:
    """`restores_cache`: the step restores a cache (guide 7); `note` explains a contract that
    `annotate` changed after inference (an opaque cache or an observed contract)."""

    contract: Contract
    commands: tuple[CommandReport, ...]
    top_reason: str | None = None
    restores_cache: bool = False
    note: str = ""


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _norm_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    return str(v)


def _pairs(m: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(m, Mapping):
        return ()
    return tuple(sorted((str(k), _norm_value(v)) for k, v in m.items()))


def _strip_jsonc(text: str) -> str:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _load_json(path: Path) -> object | None:
    try:
        return json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def _flag_value(words: Sequence[str], flags: Sequence[str]) -> str | None:
    for k, w in enumerate(words):
        for f in flags:
            if w == f and k + 1 < len(words):
                return words[k + 1]
            if w.startswith(f + "="):
                return w[len(f) + 1:]
    return None


def _has_flag(words: Sequence[str], flags: Sequence[str]) -> bool:
    for w in words:
        if w == "--":
            return False
        for f in flags:
            if w == f or w.startswith(f + "="):
                return True
    return False


def _positionals(words: Sequence[str]) -> list[str]:
    return [w for w in words if not w.startswith("-")]


def _norm_program(word: str) -> str:
    w = word.replace("\\", "/")
    if "node_modules/.bin/" in w:
        return w.rsplit("/", 1)[1]
    return w


@dataclass
class _Acc:
    reads: set[Region] = field(default_factory=set)
    writes: set[Region] = field(default_factory=set)
    out_paths: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)
    commands: list[CommandReport] = field(default_factory=list)
    top: str | None = None
    restores_cache: bool = False
    out_unknown: bool = False


class _Inference:
    def __init__(self, step: Step, job: JobInstance, repo_root: Path,
                 workflow_env: Mapping[str, str], workflow_defaults: object) -> None:
        self.step = step
        self.job = job
        self.root = Path(repo_root)
        self.matrix = dict(job.matrix)
        self.env_names = set(workflow_env) | {k for k, _ in job.env}
        self.workflow_defaults = workflow_defaults
        self.acc = _Acc()
        self._pkg: object = ...
        self._berry: bool | None = None
        self.index = self._step_index()

    # -- bookkeeping ---------------------------------------------------------

    def _step_index(self) -> int:
        for k, s in enumerate(self.job.steps):
            if s is self.step:
                return k
        for k, s in enumerate(self.job.steps):
            if s == self.step:
                return k
        return len(self.job.steps)

    def top(self, origin: str, text: str, reason: str) -> None:
        self.acc.commands.append(CommandReport(origin, text, "top", reason))
        if self.acc.top is None:
            self.acc.top = f"{reason} [{origin}: {text}]" if text else f"{reason} [{origin}]"

    def record(self, origin: str, text: str, template: str, detail: str = "") -> None:
        self.acc.commands.append(CommandReport(origin, text, template, detail))
        self.acc.templates.append(template)

    # -- repository files ----------------------------------------------------

    def pkg(self) -> Mapping | None:
        if self._pkg is ...:
            data = _load_json(self.root / "package.json")
            self._pkg = data if isinstance(data, Mapping) else None
        return self._pkg  # type: ignore[return-value]

    def scripts(self) -> Mapping[str, str] | None:
        pkg = self.pkg()
        if pkg is None:
            return None
        s = pkg.get("scripts")
        if not isinstance(s, Mapping):
            return {}
        return {str(k): str(v) for k, v in s.items() if isinstance(v, str)}

    def deps(self) -> frozenset[str]:
        pkg = self.pkg() or {}
        names: set[str] = set()
        for key in ("dependencies", "devDependencies", "optionalDependencies", "peerDependencies"):
            d = pkg.get(key)
            if isinstance(d, Mapping):
                names |= {str(k) for k in d}
        return frozenset(names)

    def yarn_berry(self) -> bool:
        if self._berry is None:
            pm = str((self.pkg() or {}).get("packageManager", ""))
            m = re.match(r"yarn@(\d+)", pm)
            self._berry = (self.root / ".yarnrc.yml").is_file() or bool(m and int(m.group(1)) >= 2)
        return self._berry

    def tsconfig(self, rel: str) -> dict[str, object]:
        """compilerOptions `outDir` (resolved to a repository-relative path) and `noEmit`,
        following relative `extends` (depth-limited)."""
        found: dict[str, object] = {}

        def visit(path: str, depth: int) -> None:
            if depth > MAX_DEPTH:
                return
            file = self.root / path
            if file.is_dir():
                file = file / "tsconfig.json"
                path = posixpath.join(path, "tsconfig.json")
            data = _load_json(file)
            if not isinstance(data, Mapping):
                return
            opts = data.get("compilerOptions")
            if isinstance(opts, Mapping):
                if "outDir" in opts and "outDir" not in found:
                    found["outDir"] = posixpath.normpath(
                        posixpath.join(posixpath.dirname(path), str(opts["outDir"])))
                if "noEmit" in opts and "noEmit" not in found:
                    found["noEmit"] = opts["noEmit"] is True
            ext = data.get("extends")
            bases = ext if isinstance(ext, list) else [ext] if isinstance(ext, str) else []
            for base in reversed(bases):  # later entries override earlier ones
                b = str(base)
                if not b.startswith("."):
                    continue  # package configs are not available offline
                target = posixpath.normpath(posixpath.join(posixpath.dirname(path), b))
                if not target.endswith(".json") and not (self.root / target).is_dir():
                    target += ".json"
                visit(target, depth + 1)

        visit(posixpath.normpath(rel), 0)
        return found

    # -- regions -------------------------------------------------------------

    def add_ctx_expr(self, body: str, origin: str, internal: bool) -> None:
        refs, funcs = expression_refs(body)
        if "hashFiles" in funcs:
            self.acc.reads |= {Region(SRC), Region(OUT)}
        for ref in refs:
            self.add_ref(ref, origin, internal)

    def add_ctx_text(self, text: str, origin: str, internal: bool = False) -> None:
        for body in _expr_bodies(text):
            self.add_ctx_expr(body, origin, internal)

    def add_ref(self, ref: tuple[str, ...], origin: str, internal: bool) -> None:
        ctx, rest = ref[0], ref[1:]
        reads = self.acc.reads
        text = ".".join(ref)
        if ctx == "matrix":
            if rest:
                reads.add(Region(CTX, None, f"matrix.{rest[0]}"))
            else:
                reads |= {Region(CTX, None, f"matrix.{k}") for k in sorted(self.matrix)}
        elif ctx == "inputs":
            if internal:
                return
            if rest:
                reads.add(Region(CTX, None, f"inputs.{rest[0]}"))
            else:
                self.top(origin, text, "reference to the whole inputs context")
        elif ctx in ("env", "vars", "secrets"):
            if rest:
                reads.add(Region(CTX, None, f"{ctx}.{rest[0]}"))
            else:
                self.top(origin, text, f"reference to the whole {ctx} context")
        elif ctx == "github":
            if not rest:
                self.top(origin, text, "reference to the whole github context")
            elif rest[0] in _IDENTITY_GITHUB_KEYS:
                reads.add(JOB_IDENTITY)
            else:
                reads.add(Region(CTX, None, "github." + ".".join(rest)))
        elif ctx == "strategy":
            reads.add(JOB_IDENTITY)
        elif ctx == "runner":
            reads.add(JOB_IDENTITY if rest[:1] == ("name",) else PLATFORM)
        elif ctx == "steps":
            if internal:
                return
            idx = None
            if rest:
                for k, s in enumerate(self.job.steps[: self.index]):
                    if s.binding.step_id == rest[0]:
                        idx = k
            if idx is None:
                self.top(origin, text, "reference to an unknown or later step")
            else:
                reads.add(Region(ST_STEP, None, str(idx)))
        elif ctx == "needs":
            inst = None
            if rest:
                cands = sorted(n for n in self.job.needs
                               if n == rest[0] or n.startswith(rest[0] + "["))
                inst = cands[0] if len(cands) == 1 else None
            if inst is None or len(rest) < 2:
                self.top(origin, text, "needs reference not resolved to one instance")
            elif rest[1] == "outputs" and len(rest) >= 3:
                reads.add(Region(OUTPUT, inst, rest[2]))
            elif rest[1] == "result":
                reads.add(Region(ST_JOB, inst, None))
            else:
                self.top(origin, text, "unsupported needs reference")
        else:
            self.top(origin, text, f"unsupported context {ctx}")

    _ENV_USE = (
        re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"process\.env\.([A-Za-z_][A-Za-z0-9_]*)"),
        re.compile(r"process\.env\[\s*['\"]([^'\"]+)['\"]\s*\]"),
        re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%"),
        re.compile(r"\$env:([A-Za-z_][A-Za-z0-9_]*)", re.I),
    )

    def add_env_text(self, text: str, origin: str, shadow: frozenset[str]) -> None:
        """Guide 3.3 and 3.4 on resolved command text."""
        names: set[str] = set()
        for pat in self._ENV_USE:
            names |= set(pat.findall(text))
        for name in sorted(names):
            if name in shadow:
                continue
            if name in self.env_names:
                self.acc.reads.add(Region(CTX, None, f"env.{name}"))
            elif name in _IDENTITY_ENV:
                self.acc.reads.add(JOB_IDENTITY)
            elif name in _FILE_COMMAND_ENV:
                self.top(origin, name, "workflow file command variable")
            elif name == "GITHUB_EVENT_PATH":
                self.acc.reads.add(Region(CTX, None, "github.event"))
            elif name.startswith("GITHUB_"):
                self.acc.reads.add(Region(CTX, None, "github." + name[len("GITHUB_"):].lower()))
            elif name.startswith("RUNNER_"):
                self.acc.reads.add(PLATFORM)

    def spec_regions(self, spec: str, services: Sequence[str], with_: Mapping[str, str],
                     origin: str) -> set[Region]:
        if spec in (SRC, DEP, OUT, TOOL):
            return {Region(spec)}
        if spec == "ext:{service}":
            return {Region(EXT, None, s) for s in services}
        if spec == "run:{with.name}":
            name = self.artifact_name(with_.get("name", "artifact"), origin)
            return set() if name is None else {Region(RUN, None, name)}
        if spec == "kinds:{with.path}":
            kinds: set[str] = set()
            outs = self.root_out_dirs()
            for line in with_.get("path", "").splitlines():
                line = line.strip()
                if line and not line.startswith("!"):
                    kinds |= path_kinds(line, outs)
            return {Region(k) for k in sorted(kinds)}
        if spec == "target:{with.path}":
            # A download writes `out`, or `tool` when its target is outside the workspace.
            target = with_.get("path") or "."
            return {Region(TOOL)} if path_kinds(target) == frozenset({TOOL}) else {Region(OUT)}
        kind, _, key = spec.partition(":")
        return {Region(kind, None, key or None)}

    def artifact_name(self, raw: str, origin: str) -> str | None:
        name = _substitute(_MATRIX_SIMPLE, raw, self.matrix)
        for body in _expr_bodies(name):
            refs, _ = expression_refs(body)
            if any(r[0] in ("matrix", "strategy") or r[:2] in (("github", "job"), ("runner", "name"))
                   for r in refs):
                self.top(origin, raw, "artifact name depends on the job instance")
                return None
        return name

    def root_out_dirs(self) -> tuple[str, ...]:
        out = self.tsconfig("tsconfig.json").get("outDir")
        return (str(out),) if out else ()

    # -- templates -----------------------------------------------------------

    def apply(self, template: str, origin: str, text: str, *, services: Sequence[str] = (),
              words: Sequence[str] = (), with_: Mapping[str, str] | None = None,
              extra_writes: Sequence[str] = (), out_paths: Sequence[str] = (),
              detail: str = "") -> None:
        tpl = TEMPLATE_BY_NAME[template]
        w = with_ or {}
        for spec in tpl.reads:
            self.acc.reads |= self.spec_regions(spec, services, w, origin)
        written: set[Region] = set()
        for spec in tpl.writes + tuple(extra_writes):
            written |= self.spec_regions(spec, services, w, origin)
        for flag, spec in tpl.flag_writes:
            if _has_flag(words[1:], (flag,)):
                written |= self.spec_regions(spec, services, w, origin)
        self.acc.writes |= written
        if Region(OUT) in written and not out_paths and template != "checkout":
            self.acc.out_unknown = True  # writes `out` somewhere not declared
        self.acc.out_paths.extend(out_paths)
        self.record(origin, text, template, detail)

    # -- steps ---------------------------------------------------------------

    def run_step(self) -> None:
        b = self.step.binding
        origin = f"step:{self.index}"
        with_ = dict(b.with_)
        for _, v in b.env:
            self.add_ctx_text(v, origin)
        if b.working_directory:
            self.add_ctx_text(b.working_directory, origin)
        shadow = frozenset(k for k, _ in b.env)
        if b.kind == "run":
            self.add_ctx_text(b.text, origin)
            wd = b.working_directory or self.default_working_directory()
            self.run_script(b.text, b.shell, wd, origin, shadow)
        elif b.kind == "uses":
            self.uses(b.text, with_, origin, depth=0, stack=(), internal=False)
        else:
            self.top(origin, b.text, f"unknown binding kind {b.kind}")

    def default_working_directory(self) -> str | None:
        for raw in ((self.job.raw or {}), {"defaults": self.workflow_defaults}):
            d = raw.get("defaults") if isinstance(raw, Mapping) else None
            r = d.get("run") if isinstance(d, Mapping) else None
            if isinstance(r, Mapping) and r.get("working-directory") is not None:
                return str(r["working-directory"])
        return None

    def run_script(self, text: str, shell: str | None, wd: str | None, origin: str,
                   shadow: frozenset[str]) -> None:
        if shell is not None:
            base = shell.split()[0] if shell.split() else ""
            if base not in ("bash", "sh", "pwsh", "powershell", "cmd"):
                self.top(origin, shell, "unsupported shell")
                return
        cwd = _norm_dir(wd) if wd else "."
        script = _substitute(_MATRIX_SIMPLE, text, self.matrix)
        self.add_env_text(script, origin, shadow)
        self.script_body(script, cwd, origin, depth=0, stack=(), shadow=shadow)

    def script_body(self, script: str, cwd: str | None, origin: str, depth: int,
                    stack: tuple[str, ...], shadow: frozenset[str]) -> None:
        commands = split_commands(script)
        if not commands:
            self.record(origin, script.strip(), "noop", "empty script")
        for cmd in commands:
            cwd = self.command(cmd, cwd, origin, depth, stack, shadow)

    # -- commands ------------------------------------------------------------

    def command(self, cmd: ShellCommand, cwd: str | None, origin: str, depth: int,
                stack: tuple[str, ...], shadow: frozenset[str]) -> str | None:
        if cmd.problem:
            self.top(origin, cmd.text, cmd.problem)
            return cwd
        words = list(cmd.words)
        while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
            words.pop(0)
        if not words:
            self.record(origin, cmd.text, "noop", "variable assignment")
            return cwd
        words[0] = _norm_program(words[0])
        return self.words(words, cmd.text, cwd, origin, depth, stack, shadow)

    def words(self, words: list[str], text: str, cwd: str | None, origin: str, depth: int,
              stack: tuple[str, ...], shadow: frozenset[str]) -> str | None:
        prog = words[0]
        if prog in NOOP_PROGRAMS:
            if prog in ("echo", "printf") and any(
                    c in w for w in words[1:] for c in _STATEFUL_WORKFLOW_COMMANDS):
                self.top(origin, text, "workflow command changes state")
                return cwd
            self.record(origin, text, "noop")
            if prog == "cd":
                return _cd(cwd, words[1:])
            return cwd
        if self.runner(words, text, cwd, origin, depth, stack, shadow):
            return cwd
        row = self.match_row(words)
        if row is not None:
            self.command_row(row, words, text, cwd, origin, depth, stack, shadow)
            return cwd
        if self.file_utility(words, text, cwd, origin):
            return cwd
        inv = _script_invocation(words)
        if inv is not None:
            name, args, if_present, problem = inv
            if problem:
                self.top(origin, text, problem)
            else:
                self.resolve_script(name, args, if_present, text, cwd, origin, depth, stack,
                                    shadow)
            return cwd
        self.top(origin, text, "unknown command")
        return cwd

    # -- file utilities (guide 4.5) -------------------------------------------

    def util_kinds(self, path: str, base: str | None) -> frozenset[str] | None:
        """Job-local kinds a file-utility path argument covers; None (top) for an unknown or
        variable path. `base` is the repository-relative directory the path is relative to
        (None: unknown). A glob maps by its fixed prefix; a negated pattern (`!x`) covers
        nothing."""
        p = path.strip()
        if not p:
            return None
        if p.startswith("!"):
            return frozenset()
        for prefix in _WORKSPACE_PREFIXES:
            if p == prefix or p.startswith(prefix + "/"):
                p, base = "." + p[len(prefix):], "."
                break
        else:
            for prefix in _OUTSIDE_PREFIXES:
                if p == prefix or p.startswith(prefix + "/"):
                    return frozenset({DEP}) if "node_modules" in p.split("/") else frozenset({TOOL})
        if "$" in p or "`" in p:
            return None
        if p.startswith("/"):
            return frozenset({TOOL}) if p == "/tmp" or p.startswith("/tmp/") else None
        if base is None:
            return None
        full = posixpath.normpath(posixpath.join(base, p))
        if full == ".." or full.startswith("../"):
            return None
        fixed: list[str] = []
        for part in (x for x in full.split("/") if x not in ("", ".")):
            if any(c in _GLOB_CHARS for c in part):
                break
            fixed.append(part)
        return path_kinds("/".join(fixed) or ".", self.root_out_dirs())

    def _util_out_path(self, path: str, base: str | None) -> str | None:
        """Repository-relative fixed prefix of a path argument that maps to `out` only."""
        p = path.strip()
        for prefix in _WORKSPACE_PREFIXES:
            if p == prefix or p.startswith(prefix + "/"):
                p, base = "." + p[len(prefix):], "."
        if base is None or "$" in p or p.startswith("/"):
            return None
        fixed: list[str] = []
        for part in posixpath.normpath(posixpath.join(base, p)).split("/"):
            if any(c in _GLOB_CHARS for c in part):
                break
            if part not in ("", "."):
                fixed.append(part)
        return "/".join(fixed) or None

    def file_utility(self, words: list[str], text: str, cwd: str | None, origin: str) -> bool:
        """Guide 4.5. Returns False when no row applies (the command falls through)."""
        prog = words[0]
        row = next((r for r in FILE_UTILITY_ROWS if r.program == prog and (
            not r.sub or (len(words) > 1 and words[1] == r.sub))), None)
        if row is None:
            return False
        args = list(words[2:] if row.sub else words[1:])
        if _has_flag(args, row.top_flags):
            self.top(origin, text, f"unsupported {prog} flag")
            return True
        if row.role == "tar":
            parsed = _parse_tar(args)
            if isinstance(parsed, str):
                self.top(origin, text, parsed)
                return True
            reads = [(_tar_base(cwd, d), x) for d, x in parsed[0]]
            writes = [(_tar_base(cwd, d), x) for d, x in parsed[1]]
        else:
            pos, flags, values, read_values = _util_args(args, row.value_flags, row.read_flags)
            got = _util_roles(row, pos, flags, values, read_values)
            if got is None:
                return False
            if isinstance(got, str):
                self.top(origin, text, got)
                return True
            reads = [(cwd, x) for x in got[0]]
            writes = [(cwd, x) for x in got[1]]
        read_kinds: set[str] = set()
        write_kinds: set[str] = set()
        out_paths: list[str] = []
        for base, path in reads:
            kinds = self.util_kinds(path, base)
            if kinds is None:
                self.top(origin, text, f"unknown or variable path {path!r}")
                return True
            read_kinds |= kinds
        for base, path in writes:
            kinds = self.util_kinds(path, base)
            if kinds is None:
                self.top(origin, text, f"unknown or variable path {path!r}")
                return True
            write_kinds |= kinds
            if OUT in kinds:  # everything it writes into `out` lies under the fixed prefix
                out_paths.append(self._util_out_path(path, base) or ".")
        self.acc.reads |= {Region(k) for k in read_kinds}
        self.acc.writes |= {Region(k) for k in write_kinds}
        self.acc.out_paths.extend(out_paths)
        self.record(origin, text, "file-util",
                    f"reads {sorted(read_kinds)} writes {sorted(write_kinds)}")
        return True

    def runner(self, words: list[str], text: str, cwd: str | None, origin: str, depth: int,
               stack: tuple[str, ...], shadow: frozenset[str]) -> bool:
        for row in RUNNER_ROWS:
            k = len(row.prefix)
            if tuple(words[:k]) != row.prefix:
                continue
            rest = words[k:]
            package: str | None = None
            j = 0
            while j < len(rest) and rest[j].startswith("-"):
                w = rest[j]
                if w == "--":
                    j += 1
                    break
                if _has_flag([w], row.top_flags):
                    self.top(origin, text, f"unsupported runner flag {w}")
                    return True
                flag = w.split("=", 1)[0]
                if flag in row.value_flags:
                    val = w.split("=", 1)[1] if "=" in w else (rest[j + 1] if j + 1 < len(rest) else "")
                    if flag in row.package_flags:
                        package = val
                    j += 1 if "=" in w else 2
                    continue
                j += 1
            inner = rest[j:]
            if not inner:
                self.top(origin, text, "runner without a command")
                return True
            spec = package or inner[0]
            pkg_name = _package_name(spec)
            bin_name = _PACKAGE_BINS.get(pkg_name, pkg_name.rsplit("/", 1)[-1]) \
                if package is None else inner[0]
            versioned = spec != pkg_name  # an explicit version may differ from the local one
            local = not versioned and (pkg_name in self.deps() or bool(
                set(_BIN_PACKAGES.get(bin_name, ())) & self.deps()))
            download = row.mode == "download" or (row.mode == "npx" and not local)
            if download:
                self.apply("install-unpinned", origin, text, detail=f"downloads {spec}")
            self.words([bin_name] + inner[1:], " ".join([bin_name] + inner[1:]), cwd,
                       origin, depth, stack, shadow)
            return True
        return False

    def match_row(self, words: Sequence[str]) -> CommandRow | None:
        prog, rest = words[0], list(words[1:])
        if "--" in rest:
            rest = rest[: rest.index("--")]
        pos = _positionals(rest)
        for row in COMMAND_ROWS:
            if row.program != prog:
                continue
            after = pos
            if row.sub:
                matched = None
                for s in row.sub:
                    parts = s.split() if s else []
                    if (not parts and not pos) or (parts and pos[: len(parts)] == parts):
                        matched = parts
                        break
                if matched is None:
                    continue
                after = pos[len(matched):]
            if row.any_flags and not _has_flag(rest, row.any_flags):
                continue
            if row.no_flags and _has_flag(rest, row.no_flags):
                continue
            if row.positional == "none" and after:
                continue
            if row.positional == "some" and not after:
                continue
            if row.no_positional and set(pos) & set(row.no_positional):
                continue
            if row.when == "yarn-berry" and not self.yarn_berry():
                continue
            if row.when == "yarn-classic" and self.yarn_berry():
                continue
            return row
        return None

    def command_row(self, row: CommandRow, words: list[str], text: str, cwd: str | None,
                    origin: str, depth: int, stack: tuple[str, ...],
                    shadow: frozenset[str]) -> None:
        template = row.template
        out_paths: tuple[str, ...] = ()
        detail = ""
        if TEMPLATE_BY_NAME[template].out_paths == "build" or row.tsconfig:
            explicit = _flag_value(words[1:], row.out_flags) if row.out_flags else None
            cfg: dict[str, object] = {}
            if cwd is not None:
                if row.tsconfig:
                    project = _flag_value(words[1:], ("-p", "--project"))
                    if project is None and _has_flag(words[1:], ("-b", "--build")):
                        pos = _positionals(words[1:])
                        project = pos[0] if pos else None
                    cfg = self.tsconfig(posixpath.join(cwd, project or "tsconfig.json"))
                else:
                    cfg = self.tsconfig(posixpath.join(cwd, "tsconfig.json"))
            if row.tsconfig and cfg.get("noEmit") is True:
                template = "typecheck"
                detail = "tsconfig sets noEmit"
            elif TEMPLATE_BY_NAME[template].out_paths == "build":
                if cwd is None:
                    detail = "unknown working directory: output not transferable"
                elif explicit is not None:
                    out_paths = _out_path(cwd, explicit)
                elif cfg.get("outDir"):
                    out_paths = _out_path(".", str(cfg["outDir"]))
                else:
                    out_paths = _out_path(cwd, "dist")
        self.apply(template, origin, text, services=row.services, words=words,
                   extra_writes=row.extra_writes, out_paths=out_paths, detail=detail)
        if row.lifecycle and not _has_flag(words[1:], ("--ignore-scripts",)):
            self.install_lifecycle(text, cwd, origin, depth, stack, shadow)

    def install_lifecycle(self, text: str, cwd: str | None, origin: str, depth: int,
                          stack: tuple[str, ...], shadow: frozenset[str]) -> None:
        """Root lifecycle scripts run by an install (implementation note 6)."""
        if cwd is None:
            self.top(origin, text, "install lifecycle scripts in an unknown directory")
            return
        if cwd != ".":
            data = _load_json(self.root / cwd / "package.json")
            scripts = data.get("scripts") if isinstance(data, Mapping) else None
            if not isinstance(data, Mapping) or (
                    isinstance(scripts, Mapping) and any(h in scripts for h in _INSTALL_LIFECYCLE)):
                self.top(origin, text, "install lifecycle scripts outside the repository root")
            return
        scripts_map = self.scripts() or {}
        for h in _INSTALL_LIFECYCLE:
            if h in scripts_map:
                self.expand_script(h, (), text, origin, depth, stack, shadow)

    def resolve_script(self, name: str, args: Sequence[str], if_present: bool, text: str,
                       cwd: str | None, origin: str, depth: int, stack: tuple[str, ...],
                       shadow: frozenset[str]) -> None:
        if cwd != ".":
            self.top(origin, text, "script resolution outside the repository root (v1)")
            return
        scripts = self.scripts()
        if scripts is None:
            self.top(origin, text, "no root package.json")
            return
        if name not in scripts:
            if if_present:
                self.record(origin, text, "noop", f"missing script {name} with --if-present")
            else:
                self.top(origin, text, f"missing script {name}")
            return
        self.acc.commands.append(CommandReport(origin, text, "resolve", name))
        for hook in (f"pre{name}", name, f"post{name}"):
            if hook in scripts:
                self.expand_script(hook, args if hook == name else (), text, origin, depth,
                                   stack, shadow)

    def expand_script(self, name: str, args: Sequence[str], text: str, origin: str, depth: int,
                      stack: tuple[str, ...], shadow: frozenset[str]) -> None:
        scripts = self.scripts() or {}
        if depth + 1 > MAX_DEPTH:
            self.top(origin, text, f"script nesting deeper than {MAX_DEPTH}")
            return
        if name in stack:
            self.top(origin, text, f"cyclic script {' -> '.join(stack + (name,))}")
            return
        body = scripts[name]
        if args:
            body = body + " " + " ".join(shlex.quote(a) for a in args)
        self.add_env_text(body, f"script:{name}", shadow)
        self.script_body(body, ".", f"script:{name}", depth + 1, stack + (name,), shadow)

    # -- actions -------------------------------------------------------------

    def uses(self, ref: str, with_: Mapping[str, str], origin: str, depth: int,
             stack: tuple[str, ...], internal: bool) -> None:
        """An action step. Its `with` values contribute context reads (guide 3.2) unless the
        action is a cache action, whose inputs only name the cache (cold-cache semantics)."""
        ref = ref.strip()
        if ref.startswith("docker://"):
            self.top(origin, ref, "docker action")
            return
        if ref == "." or ref.startswith("./"):
            for key in sorted(with_):
                self.add_ctx_text(with_[key], origin, internal)
            self.composite(ref, with_, origin, depth, stack)
            return
        name, at, _version = ref.partition("@")
        if not at:
            self.top(origin, ref, "action reference without a version")
            return
        row = _match_action(name)
        if row is None:
            self.top(origin, ref, "unknown action")
            return
        for key, mode in row.top_if:
            val = with_.get(key)
            if mode == "missing" and val is None:
                self.top(origin, ref, f"`with.{key}` missing")
                return
            if val is None:
                continue
            if mode == "any" or (mode == "literal" and "${{" not in val) or (
                    mode == "truthy" and val.strip().lower() not in ("false", "")):
                self.top(origin, ref, f"`with.{key}` not covered by the template")
                return
        if row.check and not _ACTION_CHECKS[row.check](with_):
            self.top(origin, ref, f"`{row.check}` does not hold")
            return
        if row.template in ("upload-artifact", "download-artifact") and \
                with_.get("name", "").startswith(TRANSFER_PREFIX) and \
                (internal or depth > 0 or parse_transfer(self.step.binding) is None):
            self.top(origin, ref, f"artifact name uses the reserved prefix {TRANSFER_PREFIX} "
                                  "outside the transfer idiom")
            return
        if row.template != "cache":
            for key in sorted(with_):
                self.add_ctx_text(with_[key], origin, internal)
        for key, mode, spec in row.reads_if:
            val = with_.get(key)
            if val is not None and mode == "zero" and (val.strip() == "0" or "${{" in val):
                self.acc.reads |= self.spec_regions(spec, row.services, with_, origin)
        out_paths: tuple[str, ...] = ()
        if TEMPLATE_BY_NAME[row.template].out_paths == "with.path":
            lines = [x.strip() for x in with_.get("path", ".").splitlines() if x.strip()]
            dirs = [_norm_dir(x) for x in lines]
            out_paths = tuple(d for d in dirs if d is not None) if all(dirs) else ()
        self.apply(row.template, origin, ref, services=row.services, with_=with_,
                   out_paths=out_paths)
        if _restores_cache(name, _version, with_) or (
                row.template == "cache" and not name.lower().endswith("/save")):
            self.acc.restores_cache = True

    def composite(self, ref: str, with_: Mapping[str, str], origin: str, depth: int,
                  stack: tuple[str, ...]) -> None:
        rel = posixpath.normpath(ref)
        if rel.startswith(".."):
            self.top(origin, ref, "local action outside the repository")
            return
        if rel in stack:
            self.top(origin, ref, "cyclic local action")
            return
        if depth + 1 > MAX_DEPTH:
            self.top(origin, ref, f"local actions nested deeper than {MAX_DEPTH}")
            return
        base = self.root / rel
        file = base / "action.yml"
        if not file.is_file():
            file = base / "action.yaml"
        try:
            data = yaml.safe_load(file.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            self.top(origin, ref, "local action metadata not available")
            return
        runs = data.get("runs") if isinstance(data, Mapping) else None
        if not isinstance(runs, Mapping) or runs.get("using") != "composite":
            self.top(origin, ref, "local action is not a composite action")
            return
        values = dict(with_)
        inputs = data.get("inputs") if isinstance(data.get("inputs"), Mapping) else {}
        for iname in sorted(inputs):
            spec = inputs[iname]
            if iname not in values and isinstance(spec, Mapping) and "default" in spec:
                values[str(iname)] = _norm_value(spec["default"])
                self.add_ctx_text(values[str(iname)], origin)
        steps = runs.get("steps")
        if not isinstance(steps, list):
            self.top(origin, ref, "composite action without steps")
            return
        self.acc.commands.append(CommandReport(origin, ref, "resolve", "composite action"))
        for k, s in enumerate(steps):
            sub = f"action:{rel}#{k}"
            if not isinstance(s, Mapping):
                self.top(sub, "", "malformed composite step")
                continue

            def subst(v: object) -> str:
                return _substitute(_INPUTS_SIMPLE, _norm_value(v), values)

            s_with = {k2: subst(v) for k2, v in _pairs(s.get("with"))}
            s_env = {k2: subst(v) for k2, v in _pairs(s.get("env"))}
            cond = s.get("if")
            if cond is not None:
                c = _norm_value(cond)
                bodies = _expr_bodies(c) or [c]
                for body in bodies:
                    self.add_ctx_expr(_substitute(_INPUTS_SIMPLE, body, values), sub, True)
            for v in s_env.values():
                self.add_ctx_text(v, sub, internal=True)
            if "run" in s:
                run = subst(s["run"])
                self.add_ctx_text(run, sub, internal=True)
                wd = s.get("working-directory")
                wd_s = subst(wd) if wd is not None else None
                if wd_s:
                    self.add_ctx_text(wd_s, sub, internal=True)
                shell = s.get("shell")
                self.run_script(run, subst(shell) if shell is not None else None, wd_s, sub,
                                frozenset(s_env))
            elif "uses" in s:
                self.uses(subst(s["uses"]), s_with, sub, depth + 1, stack + (rel,), internal=True)
            else:
                self.top(sub, "", "composite step without run or uses")

    # -- result --------------------------------------------------------------

    def result(self) -> StepReport:
        acc = self.acc
        commands = tuple(acc.commands)
        if acc.top is not None:
            return StepReport(TOP_CONTRACT, commands, acc.top, restores_cache=acc.restores_cache)
        template = "+".join(acc.templates) if acc.templates else "noop"
        paths = () if acc.out_unknown else tuple(sorted(set(acc.out_paths)))
        contract = Contract(
            reads=frozenset(acc.reads | {PLATFORM}),
            writes=frozenset(acc.writes),
            out_paths=paths if Region(OUT) in acc.writes else (),
            template=template,
        )
        return StepReport(contract, commands, None, restores_cache=acc.restores_cache)


def _norm_dir(wd: str) -> str | None:
    w = wd.strip()
    for prefix in ("${{ github.workspace }}", "${{github.workspace}}", "$GITHUB_WORKSPACE",
                   "${GITHUB_WORKSPACE}"):
        if w.startswith(prefix):
            w = "." + w[len(prefix):]
    if "$" in w or w.startswith(("/", "~")):
        return None
    p = posixpath.normpath(w)
    return None if p.startswith("..") else p


def _cd(cwd: str | None, args: Sequence[str]) -> str | None:
    if cwd is None or len(args) != 1 or args[0] in ("-", "~"):
        return None
    return _norm_dir(posixpath.join(cwd, args[0])) if not args[0].startswith("/") else None


def _out_path(cwd: str, rel: str) -> tuple[str, ...]:
    if "$" in rel or rel.startswith(("/", "~")):
        return ()
    p = posixpath.normpath(posixpath.join(cwd, rel))
    return () if p.startswith("..") or p == "." else (p,)


def _util_args(args: Sequence[str], value_flags: Sequence[str], read_flags: Sequence[str]
               ) -> tuple[list[str], list[str], dict[str, list[str]], list[str]]:
    """(positional words, other flags, values of `value_flags`, values of `read_flags`)."""
    pos: list[str] = []
    flags: list[str] = []
    values: dict[str, list[str]] = {}
    read_values: list[str] = []
    i, end = 0, False
    while i < len(args):
        w = args[i]
        if end or not w.startswith("-") or w == "-":
            pos.append(w)
        elif w == "--":
            end = True
        else:
            name, eq, val = w.partition("=")
            if name in value_flags or name in read_flags:
                if not eq:
                    val = args[i + 1] if i + 1 < len(args) else ""
                    i += 1
                if name in read_flags:
                    read_values.append(val)
                else:
                    values.setdefault(name, []).append(val)
            else:
                flags.append(w)
        i += 1
    return pos, flags, values, read_values


def _util_roles(row: FileUtilityRow, pos: list[str], flags: list[str],
                values: Mapping[str, list[str]], read_values: list[str]
                ) -> tuple[list[str], list[str]] | str | None:
    """(source paths, target paths) of a file utility, a top reason, or None when the row
    does not apply (`sed` without `-i`, `json` without `-I -f FILE`). Values of the row's
    `read_flags` are sources."""
    target_dir = (values.get("-t") or values.get("--target-directory") or [None])[-1]
    short = [f[1:] for f in flags if f.startswith("-") and not f.startswith("--")]
    if row.role == "targets":
        return list(read_values), list(pos)
    if row.role in ("copy", "move"):
        if target_dir is not None:
            srcs, dest = list(pos), target_dir
        elif len(pos) >= 2:
            srcs, dest = pos[:-1], pos[-1]
        else:
            return f"{row.program}: missing operand"
        return srcs + read_values, [dest] + (srcs if row.role == "move" else [])
    if row.role == "link":
        symbolic = "--symbolic" in flags or any("s" in f for f in short)
        if target_dir is not None:
            targets, links = list(pos), [target_dir]
        elif len(pos) == 1:
            targets, links = list(pos), [posixpath.basename(pos[0].rstrip("/")) or "."]
        elif len(pos) >= 2:
            targets, links = pos[:-1], [pos[-1]]
        else:
            return "ln: missing operand"
        return ([] if symbolic else targets), links
    if row.role == "sed":
        # `-i`, `-i.bak`, `-Ei`: an `i` preceded only by boolean flags (after it: a suffix)
        in_place = any(f.startswith("--in-place") for f in flags) or any(
            "i" in f and set(f[:f.index("i")]) <= set("Enrsuz") for f in short)
        if not in_place:
            return None
        script_given = "-e" in values or "--expression" in values or bool(read_values)
        files = list(pos) if script_given else list(pos[1:])
        if not files:
            return None
        return files + read_values, files
    if row.role == "json":
        if ("-I" not in flags and "--in-place" not in flags) or len(read_values) != 1:
            return None
        return list(read_values), list(read_values)
    return None


_TAR_MODES = {"c": "create", "x": "extract", "t": "list", "r": "append", "u": "append"}
_TAR_LONG_MODES = {"--create": "create", "--extract": "extract", "--get": "extract",
                   "--list": "list", "--append": "append", "--update": "append"}
_TAR_VALUE_LETTERS = frozenset("fCbgKLNVX")
_TAR_TOP_LETTERS = frozenset("TFIM")  # file lists, scripts, compress programs, multi-volume
_TAR_LONG_VALUES = frozenset({"--file", "--directory", "--exclude", "--exclude-from",
                              "--owner", "--group", "--mode", "--mtime", "--format",
                              "--blocking-factor", "--label", "--newer", "--after-date",
                              "--newer-mtime", "--suffix", "--record-size", "--sort"})
_TAR_LONG_TOP = frozenset({"--to-command", "--use-compress-program", "--rsh-command",
                           "--info-script", "--new-volume-script", "--checkpoint-action",
                           "--files-from", "--transform", "--xform", "--strip-components",
                           "--delete", "--concatenate", "--catenate", "--multi-volume",
                           "--absolute-names"})

TarPath = tuple[tuple[str, ...], str]  # (the -C directories in effect, path)


def _parse_tar(args: Sequence[str]) -> tuple[list[TarPath], list[TarPath]] | str:
    """(reads, writes) of a `tar` command, each path paired with the `-C` directories in effect
    for it (applied in order from the working directory); or a top reason. Create reads the
    members and writes the archive; extract (members required) reads the archive and writes
    the members; list reads the archive; append/update reads both and writes the archive."""
    toks = list(args)
    if toks and not toks[0].startswith("-"):
        toks[0] = "-" + toks[0]  # old-style bundled letters: `tar czf out.tgz dist`
    mode: str | None = None
    archive: str | None = None
    dirs: tuple[str, ...] = ()
    members: list[TarPath] = []
    i, end = 0, False
    while i < len(toks):
        w = toks[i]
        if end or not w.startswith("-") or w == "-":
            members.append((dirs, w))
            i += 1
            continue
        if w == "--":
            end = True
            i += 1
            continue
        if w.startswith("--"):
            name, eq, val = w.partition("=")
            if name in _TAR_LONG_TOP:
                return f"unsupported tar option {name}"
            if name in _TAR_LONG_MODES:
                mode = _TAR_LONG_MODES[name]
            elif name in _TAR_LONG_VALUES:
                if not eq:
                    val = toks[i + 1] if i + 1 < len(toks) else ""
                    i += 1
                if name == "--file":
                    archive = val
                elif name == "--directory":
                    dirs = dirs + (val,)
            i += 1
            continue
        letters = w[1:]
        for k, ch in enumerate(letters):
            if ch in _TAR_TOP_LETTERS:
                return f"unsupported tar option -{ch}"
            if ch in _TAR_MODES:
                mode = _TAR_MODES[ch]
            elif ch in _TAR_VALUE_LETTERS:
                rest = letters[k + 1:]
                if rest:
                    val = rest
                else:
                    val = toks[i + 1] if i + 1 < len(toks) else ""
                    i += 1
                if ch == "f":
                    archive = val
                elif ch == "C":
                    dirs = dirs + (val,)
                break
        i += 1
    if mode is None:
        return "tar without a known mode"
    if archive is None:
        return "tar without an archive (-f)"
    arch: list[TarPath] = [] if archive == "-" else [((), archive)]
    if mode == "create":
        return (members, arch) if members else "tar -c without members"
    if mode == "extract":
        return (arch, members) if members else "tar -x without members extracts unknown paths"
    if mode == "list":
        return arch, []
    return members + arch, arch


def _tar_base(cwd: str | None, dirs: Sequence[str]) -> str | None:
    """The repository-relative directory after applying `-C` directories to `cwd`."""
    base = cwd
    for d in dirs:
        for prefix in _WORKSPACE_PREFIXES:
            if d == prefix or d.startswith(prefix + "/"):
                base = _norm_dir("." + d[len(prefix):])
                break
        else:
            if base is None or "$" in d or "`" in d or d.startswith(("/", "~")):
                return None
            base = _norm_dir(posixpath.join(base, d))
        if base is None:
            return None
    return base


def _package_name(spec: str) -> str:
    if spec.startswith("@"):
        scope, _, rest = spec.partition("/")
        return scope + "/" + rest.split("@", 1)[0]
    return spec.split("@", 1)[0]


def _match_action(name: str) -> ActionRow | None:
    low = name.lower()
    for row in ACTION_ROWS:
        pat = row.ref.lower()
        if pat.endswith("/*"):
            if low.split("/", 1)[0] == pat[:-2]:
                return row
        elif low == pat:
            return row
    return None


def _major(version: str) -> int | None:
    m = re.match(r"v?(\d+)(?:\.|\Z)", version.strip())
    return int(m.group(1)) if m else None


def _restores_cache(name: str, version: str, with_: Mapping[str, str]) -> bool:
    """Does a setup action restore a dependency cache (guide 7)? `cache` input set and not
    `false`; `actions/setup-go` also by default from v4 on (or when the version is not a
    tag, e.g. a pinned commit)."""
    low = name.lower()
    if low not in ("actions/setup-node", "actions/setup-python", "actions/setup-java",
                   "actions/setup-go", "denoland/setup-deno"):
        return False
    val = with_.get("cache")
    if val is not None:
        return val.strip().lower() not in ("", "false")
    if low == "actions/setup-go":
        major = _major(version)
        return major is None or major >= 4
    return False


_LOG_STATEMENT = re.compile(
    r"^\s*(?:console\.(?:log|info|warn|error|debug)|core\.(?:info|debug|notice|warning|error|"
    r"startGroup|endGroup))\s*\((?:[^()`]|\([^()`]*\))*\)\s*;?\s*$")


def _script_only_logs(with_: Mapping[str, str]) -> bool:
    script = with_.get("script", "")
    lines = [ln for ln in (x.split("//", 1)[0].strip() for x in script.splitlines()) if ln]
    return bool(lines) and all(_LOG_STATEMENT.match(ln) for ln in lines)


_ACTION_CHECKS = {"script-only-logs": _script_only_logs}


def _script_invocation(words: Sequence[str]) -> tuple[str, tuple[str, ...], bool, str | None] | None:
    """(script name, args, if_present, problem) for a package-manager script run, else None."""
    prog = words[0]
    rest = list(words[1:])
    if prog not in ("npm", "pnpm", "yarn", "bun", "nr"):
        return None
    dashdash = rest.index("--") if "--" in rest else len(rest)
    head = rest[:dashdash]
    if_present = "--if-present" in head
    if prog == "nr":
        pos = [k for k, w in enumerate(head) if not w.startswith("-")]
        if not pos:
            return None
        k = pos[0]
        args = tuple(w for w in rest[k + 1:] if w != "--")
        return rest[k], args, if_present, None
    # position of the subcommand (first positional word)
    k = next((j for j, w in enumerate(head) if not w.startswith("-")), None)
    if k is None:
        return None
    before = head[:k]
    problem = None
    if _has_flag(before, _PM_DIR_FLAGS):
        problem = "package-manager directory or workspace flag (v1 resolves the root only)"
    sub = head[k]
    if prog == "npm":
        if _has_flag(head, _PM_DIR_FLAGS):
            problem = "package-manager directory or workspace flag (v1 resolves the root only)"
        args = tuple(rest[dashdash + 1:])
        if sub in _NPM_RUN:
            names = [w for w in head[k + 1:] if not w.startswith("-")]
            if not names:
                return None
            return names[0], args, if_present, problem
        if sub in _NPM_LIFECYCLE:
            return _NPM_LIFECYCLE[sub], args, if_present, problem
        return None
    if prog == "bun":
        if sub != "run":
            return None
        names = [j for j in range(k + 1, len(rest)) if not rest[j].startswith("-")]
        if not names:
            return None
        j = names[0]
        return rest[j], _pm_args(rest[j + 1:]), if_present, problem
    builtins = _PNPM_BUILTINS if prog == "pnpm" else _YARN_BUILTINS
    if sub == "run":
        names = [j for j in range(k + 1, len(rest)) if not rest[j].startswith("-")]
        if not names:
            return None
        j = names[0]
        return rest[j], _pm_args(rest[j + 1:]), if_present, problem
    if sub in _PM_SCRIPT_ALIASES:
        return _PM_SCRIPT_ALIASES[sub], _pm_args(rest[k + 1:]), if_present, problem
    if sub in builtins:
        return None
    return sub, _pm_args(rest[k + 1:]), if_present, problem


def _pm_args(args: Sequence[str]) -> tuple[str, ...]:
    a = list(args)
    if a and a[0] == "--":
        a = a[1:]
    return tuple(w for w in a if w != "--if-present")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _env_mapping(env: object) -> dict[str, str]:
    if isinstance(env, Mapping):
        return {str(k): _norm_value(v) for k, v in env.items()}
    if isinstance(env, (tuple, list)):
        return {str(k): str(v) for k, v in env}
    return {}


# ---------------------------------------------------------------------------
# Observed contracts (guide 8)
# ---------------------------------------------------------------------------

TRACE_FORMAT = "ghav-trace/1"


@dataclass(frozen=True)
class ObservedStep:
    """One step of a trace file: the binding key it applies to, the paths the step was
    observed to write (create, modify or delete), and external objects it writes."""

    key: tuple
    writes: tuple[str, ...]
    external: tuple[str, ...] = ()
    source: str = ""


def trace_path(repo_root: Path, source: str) -> Path:
    """`resources/traces/<owner>__<repo>/<sha>/<benchmark-id>.json` for a snapshot at
    `resources/repos/<owner>__<repo>/<sha>` and a workflow path (benchmark id:
    `<owner>__<repo>__<workflow-file-stem>`)."""
    root = Path(repo_root)
    owner_repo, sha = root.parent.name, root.name
    stem = posixpath.splitext(posixpath.basename(source.replace("\\", "/")))[0]
    return root.parent.parent.parent / "traces" / owner_repo / sha / f"{owner_repo}__{stem}.json"


def _binding_key(entry: Mapping) -> tuple:
    b = entry.get("binding")
    if not isinstance(b, Mapping) or b.get("kind") not in ("run", "uses"):
        raise ValueError("trace step without a binding of kind run or uses")

    def pairs(m: object) -> tuple[tuple[str, str], ...]:
        if m is None:
            return ()
        if not isinstance(m, Mapping):
            raise ValueError("binding with/env must be mappings")
        return tuple(sorted((str(k), _norm_value(v)) for k, v in m.items()))

    wd, sh = b.get("working_directory"), b.get("shell")
    return (str(b["kind"]), str(b.get("text", "")), pairs(b.get("with")), pairs(b.get("env")),
            None if wd is None else str(wd), None if sh is None else str(sh))


def load_observed(path: Path) -> dict[tuple, ObservedStep]:
    """Observed contracts of a trace file keyed by binding key; empty if the file does not
    exist. Raises ValueError on a malformed file (a trace is committed input, never guessed)."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, Mapping) or data.get("format") != TRACE_FORMAT:
        raise ValueError(f"{path}: not a {TRACE_FORMAT} trace")
    steps = data.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"{path}: `steps` must be a list")
    out: dict[tuple, ObservedStep] = {}
    for entry in steps:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{path}: malformed step entry")
        key = _binding_key(entry)
        writes = [str(x) for x in (entry.get("writes") or [])] + \
            [str(x) for x in (entry.get("deletes") or [])]
        ext = [str(x) for x in (entry.get("external") or [])]
        prev = out.get(key)
        if prev is not None:  # several observations of one binding: union of their writes
            writes += list(prev.writes)
            ext += list(prev.external)
        out[key] = ObservedStep(key, tuple(sorted(set(writes))), tuple(sorted(set(ext))),
                                path.as_posix())
    return out


def _observed_arg(program: Program, repo_root: Path, observed: object) -> Mapping[tuple, ObservedStep]:
    if observed is None:
        return load_observed(trace_path(repo_root, program.source)) if program.source else {}
    if isinstance(observed, (str, Path)):
        return load_observed(Path(observed))
    if isinstance(observed, Mapping):
        return observed  # type: ignore[return-value]
    raise TypeError("observed must be None, a trace path, or a mapping")


def _observed_out_path(path: str, outs: Sequence[str]) -> str:
    parts = [x for x in posixpath.normpath(path).split("/") if x not in ("", ".")]
    for k in range(1, len(parts) + 1):
        if parts[k - 1] in outs or "/".join(parts[:k]) in outs:
            return "/".join(parts[:k])
    return "/".join(parts) or "."


def _observed_report(inf: "_Inference", entry: ObservedStep) -> StepReport | None:
    """The observed contract of `inf.step` (guide 8), or None if the trace cannot be mapped."""
    b = inf.step.binding
    inf.acc = _Acc()
    origin = f"step:{inf.index}"
    texts = [b.text] if b.kind == "run" else []
    texts += [v for _, v in b.with_] + [v for _, v in b.env]
    if b.working_directory:
        texts.append(b.working_directory)
    for t in texts:
        inf.add_ctx_text(t, origin)
    resolved = _substitute(_MATRIX_SIMPLE, b.text, inf.matrix) if b.kind == "run" else \
        "\n".join(v for _, v in b.with_)
    inf.add_env_text(resolved, origin, frozenset(k for k, _ in b.env))
    if inf.acc.top is not None:
        return None
    outs = tuple(_OUT_DIR_NAMES) + inf.root_out_dirs()
    writes: set[Region] = set()
    out_paths: list[str] = []
    for path in entry.writes:
        kinds = inf.util_kinds(path, ".")
        if kinds is None or path.strip().startswith("!"):
            return None
        writes |= {Region(k) for k in kinds}
        if OUT in kinds:
            out_paths.append(_observed_out_path(path, outs))
    writes |= {Region(EXT, None, e) for e in entry.external}
    reads = inf.acc.reads | {Region(k) for k in (SRC, DEP, OUT, TOOL)} | {PLATFORM}
    contract = Contract(frozenset(reads), frozenset(writes), False,
                        tuple(sorted(set(out_paths))) if Region(OUT) in writes else (), "observed")
    cmd = CommandReport(origin, b.text, "observed", f"trace {entry.source}")
    return StepReport(contract, (cmd,), None, note=f"observed contract ({entry.source})")


# ---------------------------------------------------------------------------
# Cache transparency (guide 7)
# ---------------------------------------------------------------------------


def _may_write_dep(template: str) -> bool:
    t = TEMPLATE_BY_NAME.get(template)
    if t is None:
        return True
    return any(spec == DEP or spec.startswith(("kinds:", "paths:")) for spec in t.writes)


def writes_dep_after_restore(c: Contract) -> bool:
    """Does a task break cache transparency when it runs after a cache restore in its job?
    Yes if it is top or writes `dep` through a command other than a pinned install (whose
    `dep` is fixed by the lockfile, so it is the same after a hit and after a miss)."""
    if c.top:
        return True
    if not any(r.kind == DEP for r in c.writes):
        return False
    return any(t != "install-pinned" and _may_write_dep(t) for t in c.template.split("+"))


def cache_violation(job: JobInstance, index: int) -> int | None:
    """Index of the first later step of `job` that writes `dep` after the cache restore at
    `index` (see `writes_dep_after_restore`), or None."""
    for k in range(index + 1, len(job.steps)):
        if writes_dep_after_restore(job.steps[k].contract):
            return k
    return None


def _opaque_cache(inf: "_Inference", c: Contract) -> Contract | None:
    """Contract of a cache restore treated as opaque: it also reads every job-local kind and
    the context values of its inputs (the cache key), and writes `dep`, `tool` and the kinds
    of its cached paths (every job-local kind for a local composite action). None when the
    inputs make it top."""
    b = inf.step.binding
    origin = f"step:{inf.index}"
    inf.acc = _Acc()
    for _, v in b.with_ + b.env:
        inf.add_ctx_text(v, origin)
    if inf.acc.top is not None:
        return None
    local = {Region(k) for k in (SRC, DEP, OUT, TOOL)}
    writes = set(c.writes) | {Region(DEP), Region(TOOL)}
    if b.kind == "uses" and (b.text == "." or b.text.startswith("./")):
        writes |= local
    else:
        for line in dict(b.with_).get("path", "").splitlines():
            if line.strip() and not line.strip().startswith("!"):
                writes |= {Region(k) for k in path_kinds(line.strip(), inf.root_out_dirs())}
    parts = c.template.split("+")
    parts = [("cache-opaque" if t == "cache" else t) for t in parts]
    if "cache-opaque" not in parts:
        parts.append("cache-opaque")
    new_out = Region(OUT) in writes and Region(OUT) not in c.writes
    return Contract(frozenset(c.reads | inf.acc.reads | local), frozenset(writes), False,
                    () if new_out else c.out_paths, "+".join(parts))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explain(step: Step, job: JobInstance, repo_root: Path, *,
            workflow_env: object = (), workflow_defaults: object = None,
            observed: Mapping[tuple, ObservedStep] | None = None) -> StepReport:
    """Infer a step's contract and report how every command was classified. A top step whose
    binding key is in `observed` gets its observed contract (guide 8)."""
    inf = _Inference(step, job, Path(repo_root), _env_mapping(workflow_env), workflow_defaults)
    inf.run_step()
    rep = inf.result()
    if rep.contract.top and observed:
        entry = observed.get(step.binding.key())
        if entry is not None:
            obs = _observed_report(inf, entry)
            if obs is not None:
                return replace(obs, commands=rep.commands + obs.commands,
                               restores_cache=rep.restores_cache)
    return rep


def infer(step: Step, job: JobInstance, repo_root: Path, *, workflow_env: object = (),
          workflow_defaults: object = None,
          observed: Mapping[tuple, ObservedStep] | None = None) -> Contract:
    """The step's contract (guide sections 2-5 and 8); pure and deterministic.

    `workflow_env`/`workflow_defaults` carry the workflow-level `env` and `defaults`
    (guide 3.3); `annotate` passes them from the program. The cache-transparency rule
    (guide 7) needs the whole job and is applied by `annotate`.
    """
    return explain(step, job, repo_root, workflow_env=workflow_env,
                   workflow_defaults=workflow_defaults, observed=observed).contract


def explain_program(program: Program, repo_root: Path, *, observed: object = None
                    ) -> dict[tuple[str, int], StepReport]:
    """Step reports of every step of every job (frozen jobs included), after the observed
    contracts (guide 8) and the cache-transparency rule (guide 7). `observed`: None (the
    default trace file of the benchmark, `trace_path`), a trace file path, or a mapping from
    `load_observed`. Keys are (instance id, step index)."""
    obs = _observed_arg(program, repo_root, observed)
    reports: dict[tuple[str, int], StepReport] = {}
    for job in sorted(program.jobs, key=lambda j: j.id):
        reps = [explain(s, job, repo_root, workflow_env=program.env,
                        workflow_defaults=program.defaults, observed=obs) for s in job.steps]
        opaque_done: set[int] = set()
        changed = True
        while changed:  # an opaque restore writes dep itself, so iterate to a fixpoint
            changed = False
            filled = replace(job, steps=tuple(replace(s, contract=r.contract)
                                              for s, r in zip(job.steps, reps)))
            for k, rep in enumerate(reps):
                if k in opaque_done or not rep.restores_cache or rep.contract.top:
                    continue
                bad = cache_violation(filled, k)
                if bad is None:
                    continue
                later = filled.steps[bad].binding
                note = (f"cache restore is opaque: step {bad} "
                        f"({later.name or later.text.strip()[:60]!r}) writes dep after it")
                inf = _Inference(filled.steps[k], filled, Path(repo_root),
                                 _env_mapping(program.env), program.defaults)
                opaque = _opaque_cache(inf, rep.contract)
                reps[k] = replace(rep, contract=opaque, note=note) if opaque is not None else \
                    StepReport(TOP_CONTRACT, rep.commands, note, True, note)
                opaque_done.add(k)
                changed = True
        for k, rep in enumerate(reps):
            reports[(job.id, k)] = rep
    return reports


def annotate(program: Program, repo_root: Path, *, observed: object = None) -> Program:
    """Copy of `program` with every step's contract inferred (see `explain_program`).

    Steps of frozen jobs get their guide contracts too (RESEARCH_SCOPE: frozen jobs get the
    union of their steps' guide contracts, not blanket top); the semantics uses them for the
    frozen job's effect (`semantics.frozen_effect`). A round trip must pass the original
    benchmark's `observed` contracts when re-annotating an emitted workflow.
    """
    reports = explain_program(program, repo_root, observed=observed)
    jobs = []
    for job in program.jobs:
        steps = tuple(replace(s, contract=reports[(job.id, k)].contract)
                      for k, s in enumerate(job.steps))
        jobs.append(replace(job, steps=steps))
    return program.with_jobs(jobs)


def report(program: Program, repo_root: Path, *, observed: object = None) -> dict:
    """JSON-ready audit of every step's contract (for `ghav contracts` and contracts.json).
    Steps of frozen jobs are reported with their guide contracts and `frozen: true`."""
    reports = explain_program(program, repo_root, observed=observed)
    rows = []
    counts: dict[str, int] = {}
    for job in program.jobs:
        for k, s in enumerate(job.steps):
            rep = reports[(job.id, k)]
            c = rep.contract
            for t in c.template.split("+"):
                counts[t] = counts.get(t, 0) + 1
            rows.append({
                "job": job.id,
                "index": k,
                "frozen": job.frozen,
                "name": s.binding.name,
                "binding": s.binding.text,
                "template": c.template,
                "top": c.top,
                "top_reason": rep.top_reason,
                "note": rep.note,
                "reads": sorted(str(r) for r in c.reads),
                "writes": sorted(str(r) for r in c.writes),
                "out_paths": list(c.out_paths),
                "commands": [
                    {"origin": x.origin, "text": x.text, "outcome": x.outcome, "detail": x.detail}
                    for x in rep.commands
                ],
            })
    n = len(rows)
    tops = sum(1 for r in rows if r["top"])
    nf = [r for r in rows if not r["frozen"]]
    nf_tops = sum(1 for r in nf if r["top"])
    return {
        "steps": rows,
        "summary": {
            "steps": n,
            "top_steps": tops,
            "top_share": (tops / n) if n else 0.0,
            "unfrozen_steps": len(nf),
            "unfrozen_top_steps": nf_tops,
            "unfrozen_top_share": (nf_tops / len(nf)) if nf else 0.0,
            "observed_steps": sum(1 for r in rows if r["template"] == "observed"),
            "opaque_caches": sum(1 for r in rows if "cache-opaque" in r["template"].split("+")
                                 or (r["note"] or "").startswith("cache restore is opaque")),
            "template_counts": dict(sorted(counts.items())),
        },
    }


def _md_cell(values: Iterable[str]) -> str:
    vals = [f"`{v}`" for v in values]
    return ", ".join(vals) if vals else "–"


def template_table_markdown() -> str:
    """The template, action, command and runner tables as Markdown, for auditing against
    docs/CONTRACT_GUIDE.md."""
    lines = ["## Templates", "",
             "Every non-top step also reads `ctx:platform` (guide 3.1).", "",
             "| Template | Reads | Writes | Writes if flag | out_paths | Note |",
             "|---|---|---|---|---|---|"]
    for t in TEMPLATES:
        fw = ", ".join(f"`{f}` → `{r}`" for f, r in t.flag_writes) or "–"
        lines.append(f"| {t.name} | {_md_cell(t.reads)} | {_md_cell(t.writes)} | {fw} | "
                     f"{t.out_paths or '–'} | {t.note or ''} |")
    lines += ["", "## Actions", "", "| Reference | Template | Services | Top when | Check |",
              "|---|---|---|---|---|"]
    for a in ACTION_ROWS:
        top_if = ", ".join(f"`{k}` {m}" for k, m in a.top_if) or "–"
        lines.append(f"| `{a.ref}@*` | {a.template} | {_md_cell(a.services)} | {top_if} | "
                     f"{a.check or '–'} |")
    lines += ["", "## Commands", "",
              "First matching row wins.", "",
              "| Program | Sub | Any flag | No flag | Positional | When | Template | "
              "Extra writes | Lifecycle | Note |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for c in COMMAND_ROWS:
        sub = ", ".join(f"`{s}`" if s else "(none)" for s in c.sub) or "any"
        lines.append(
            f"| `{c.program}` | {sub} | {_md_cell(c.any_flags)} | {_md_cell(c.no_flags)} | "
            f"{c.positional} | {c.when or '–'} | {c.template} | {_md_cell(c.extra_writes)} | "
            f"{'yes' if c.lifecycle else '–'} | {c.note} |")
    lines += ["", "## File utilities (guide 4.5)", "",
              "| Program | Sub | Role | Value flags | Read flags | Top flags |",
              "|---|---|---|---|---|---|"]
    for f in FILE_UTILITY_ROWS:
        lines.append(f"| `{f.program}` | {f.sub or '–'} | {f.role} | {_md_cell(f.value_flags)} | "
                     f"{_md_cell(f.read_flags)} | {_md_cell(f.top_flags)} |")
    lines += ["", "## Package runners", "", "| Prefix | Mode | Top flags |", "|---|---|---|"]
    for r in RUNNER_ROWS:
        lines.append(f"| `{' '.join(r.prefix)}` | {r.mode} | {_md_cell(r.top_flags)} |")
    lines += ["", "## Shell-only commands (noop)", "", _md_cell(NOOP_PROGRAMS), "",
              "## Script resolution", "",
              f"Root `package.json` only; depth at most {MAX_DEPTH}; `pre`/`post` hooks; "
              "cycles and missing scripts give top. Install lifecycle scripts: "
              f"{_md_cell(_INSTALL_LIFECYCLE)}.", "",
              "## Implementation notes (where the guide is silent)", ""]
    lines += [f"{k}. {note}" for k, note in enumerate(IMPLEMENTATION_NOTES, 1)]
    return "\n".join(lines) + "\n"
