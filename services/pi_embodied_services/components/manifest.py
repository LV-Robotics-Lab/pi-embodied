# Copyright 2026 The pi-embodied Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A robot's primitive manifest: the one declaration of its tools and code primitives.

``packages/embodied/src/primitives/manifests/<robot>.json`` (shared entries in ``common/*.json``)
is read by pi (tool schemas, the code-mode prompt) and here (``code.api``, the whitelist a
program's calls go through, the server's startup self-check). Python finds it by repo-relative
path from this package; ``PI_EMBODIED_MANIFESTS`` points elsewhere (services/README.md).

Entry fields: ``name``; ``side`` (``env``: a facade RPC method runs it for the tool and the
program alike; ``ts``: a pi-side tool; ``code``: a code primitive without a tool); ``method`` (env,
code); ``tier`` (one of high, low, raw, privileged); ``mutating``; ``requires`` (capabilities the
run must have, e.g. ``sam3``, ``ik``, ``grasp``, ``privileged``); ``params`` (name ->
``{type, required, description, values, items, minimum, maximum, modes}``); ``doc`` (``tool`` and
``code``; the code doc's ``Example:`` section is dropped in the S4 tier); ``result`` (tool
display: ``motion`` or ``read``). ``{"use": "<file>/<name>", ...}`` pulls a shared entry from
``common/<file>.json`` and overrides the fields given.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from pi_embodied_services.components import code_api as registry

#: ``services/pi_embodied_services/components`` -> the repo root -> the manifests.
DEFAULT_DIR = (
    Path(__file__).resolve().parents[3]
    / "packages"
    / "embodied"
    / "src"
    / "primitives"
    / "manifests"
)
MANIFEST_ENV = "PI_EMBODIED_MANIFESTS"

SIDES = ("env", "ts", "code")
TIERS = ("high", "low", "raw", "privileged")
PARAM_TYPES = (
    "number",
    "integer",
    "boolean",
    "string",
    "enum",
    "vec3",
    "quat",
    "array",
    "object",
)
ENTRY_KEYS = {
    "name",
    "side",
    "method",
    "tier",
    "mutating",
    "requires",
    "params",
    "doc",
    "result",
    "module",
}
PARAM_KEYS = {
    "type",
    "required",
    "description",
    "values",
    "items",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "modes",
}
#: RPC methods every facade serves that are not primitives (the transport's and code mode's own).
FRAMEWORK_RPC = frozenset(
    {
        "code.api",
        "code.run",
        "code.helpers",
        "code.preflight",
        "code.set_limits",
        "env.get_env_meta",
        "env.reset",
    }
)


#: RPC methods a program must never reach, whatever a manifest says.
NEVER_PRIMITIVES = frozenset(
    {
        "stop",
        "cancel",
        "shutdown",
        "healthz",
        "env.reset",
        "code.run",
        "code.api",
        "code.helpers",
        "code.preflight",
        "code.set_limits",
    }
)


class ManifestError(ValueError):
    """A manifest that does not parse, or does not match its server."""


def manifest_dir() -> Path:
    return Path(os.environ.get(MANIFEST_ENV) or DEFAULT_DIR)


def _read(path: Path, files: dict[str, bytes] | None = None) -> dict:
    try:
        data = path.read_bytes()
        if files is not None:
            files[
                path.name if path.parent.name != "common" else f"common/{path.name}"
            ] = data
        return json.loads(data.decode("utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(
            f"no primitive manifest {path} (set {MANIFEST_ENV})"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: {exc}") from exc


def _common(
    ref: str, root: Path, cache: dict[str, dict], files: dict[str, bytes]
) -> dict:
    if "/" not in ref:
        raise ManifestError(f"use {ref!r}: expected '<file>/<name>'")
    file, name = ref.split("/", 1)
    if file not in cache:
        raw = _read(root / "common" / f"{file}.json", files)
        cache[file] = {e["name"]: e for e in raw["primitives"]}
        cache[file]["\0internal"] = list(raw.get("internal", []))
    try:
        return copy.deepcopy(cache[file][name])
    except KeyError:
        raise ManifestError(
            f"use {ref!r}: common/{file}.json has no {name!r}"
        ) from None


def check_entry(e: Mapping[str, Any], where: str) -> None:
    """Raise ManifestError when an entry does not follow the schema."""
    unknown = set(e) - ENTRY_KEYS
    if unknown:
        raise ManifestError(f"{where}: unknown field(s) {sorted(unknown)}")
    name = e.get("name")
    if not isinstance(name, str) or not name.isidentifier():
        raise ManifestError(f"{where}: name {name!r} is not an identifier")
    if e.get("side") not in SIDES:
        raise ManifestError(f"{where}: side must be one of {SIDES}")
    if e.get("tier") not in TIERS:
        raise ManifestError(
            f"{where}: tier must be one of {TIERS}, got {e.get('tier')!r}"
        )
    if e["side"] in ("env", "code") and not isinstance(e.get("method"), str):
        raise ManifestError(f"{where}: side {e['side']} needs an RPC method")
    if e.get("method") in NEVER_PRIMITIVES:
        raise ManifestError(f"{where}: {e['method']} is never a primitive")
    if e["side"] == "ts" and "method" in e:
        raise ManifestError(f"{where}: a ts tool has no RPC method")
    doc = e.get("doc") or {}
    if set(doc) - {"tool", "code"} or not doc:
        raise ManifestError(f"{where}: doc must have tool and/or code")
    if e["side"] == "code" and "tool" in doc:
        raise ManifestError(f"{where}: a code-side primitive has no tool doc")
    if e["side"] == "ts" and "code" in doc:
        raise ManifestError(f"{where}: a ts tool has no code doc")
    if not isinstance(e.get("requires", []), list):
        raise ManifestError(f"{where}: requires must be a list")
    for k, p in (e.get("params") or {}).items():
        if not k.isidentifier():
            raise ManifestError(f"{where}: parameter {k!r} is not an identifier")
        if set(p) - PARAM_KEYS:
            raise ManifestError(
                f"{where}.{k}: unknown field(s) {sorted(set(p) - PARAM_KEYS)}"
            )
        if p.get("type") not in PARAM_TYPES:
            raise ManifestError(f"{where}.{k}: type must be one of {PARAM_TYPES}")
        if p["type"] == "enum" and not p.get("values"):
            raise ManifestError(f"{where}.{k}: an enum needs values")


def load_manifest(robot: str, root: Path | None = None) -> dict:
    """The robot's manifest with its shared entries resolved and every entry checked."""
    root = root or manifest_dir()
    files: dict[str, bytes] = {}
    raw = _read(root / f"{robot}.json", files)
    if raw.get("robot") != robot:
        raise ManifestError(f"{robot}.json names robot {raw.get('robot')!r}")
    cache: dict[str, dict] = {}
    entries = []
    for i, e in enumerate(raw.get("primitives", [])):
        if "use" in e:
            base = _common(e["use"], root, cache, files)
            base.update({k: v for k, v in e.items() if k != "use"})
            e = base
        check_entry(e, f"{robot}.json primitives[{i}] ({e.get('name')})")
        entries.append(e)
    seen: dict[tuple[str, str], int] = {}
    for e in entries:
        key = (e["name"], "privileged" if e["tier"] == "privileged" else "open")
        if key in seen:
            raise ManifestError(f"{robot}.json declares {e['name']!r} twice")
        seen[key] = 1
    internal = list(raw.get("internal", []))
    for c in cache.values():
        internal += c.get("\0internal", [])
    return {
        "robot": robot,
        "internal": internal,
        "primitives": entries,
        "digest": digest(files),
    }


def digest(files: Mapping[str, bytes]) -> str:
    """The manifest's version: sha256 over its file and the shared files it uses, by name in
    sorted order (pi computes the same from the same bytes and refuses a server of another)."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode() + b"\0" + files[name] + b"\0")
    return h.hexdigest()


def available(entry: Mapping[str, Any], have: Callable[[str], bool]) -> bool:
    return all(have(r) for r in entry.get("requires", []))


def code_primitives(
    manifest: Mapping[str, Any], have: Callable[[str], bool]
) -> list[registry.Primitive]:
    """The manifest's code primitives this run has (``requires`` met)."""
    out = []
    for e in manifest["primitives"]:
        if e["side"] == "ts" or "code" not in e["doc"] or not available(e, have):
            continue
        params = {
            k: registry.Param(
                p["type"],
                p.get("description", ""),
                bool(p.get("required", False)),
                # "{{var}}": a list only pi knows (e.g. the arms of this task); the server's
                # method checks the value itself.
                tuple(p["values"]) if isinstance(p.get("values"), list) else (),
                p.get("minimum"),
                p.get("maximum"),
            )
            for k, p in (e.get("params") or {}).items()
            if "code" in p.get("modes", ["tool", "code"])
        }
        out.append(
            registry.Primitive(
                e["name"],
                e["method"],
                e["doc"]["code"],
                params,
                mutating=bool(e.get("mutating", False)),
                tiers=(e["tier"],),
            )
        )
    return out


def self_check(
    facade: Any, manifest: Mapping[str, Any], have: Callable[[str], bool]
) -> None:
    """Fail closed at server start: every available env/code entry has its RPC method, and every
    business RPC method is declared (by some entry, whatever its requirements) or internal."""
    rpc = facade._rpc
    missing = [
        f"{e['name']} ({e['method']})"
        for e in manifest["primitives"]
        if e["side"] != "ts" and available(e, have) and e["method"] not in rpc
    ]
    declared = {e["method"] for e in manifest["primitives"] if e["side"] != "ts"}
    internal = set(manifest.get("internal", [])) | FRAMEWORK_RPC
    extra = sorted(m for m in rpc if m not in declared and m not in internal)
    problems = []
    if missing:
        problems.append("declared but not served: " + ", ".join(missing))
    if extra:
        problems.append("served but neither declared nor internal: " + ", ".join(extra))
    for e in manifest["primitives"]:
        fn = rpc.get(e.get("method", ""))
        if e["side"] == "ts" or fn is None:
            continue
        # A tool-only parameter (modes ["tool"]) is pi's; the method never sees it.
        mismatch = _signature_mismatch(
            fn,
            {
                k: p
                for k, p in (e.get("params") or {}).items()
                if "code" in p.get("modes", ["tool", "code"])
            },
        )
        if mismatch:
            problems.append(f"{e['name']}: {mismatch}")
    if problems:
        raise ManifestError(
            f"{manifest['robot']} manifest does not match its env server: "
            + "; ".join(problems)
        )


def _signature_mismatch(fn: Callable[..., Any], params: Mapping[str, Any]) -> str:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ""
    ps = sig.parameters
    if any(p.kind is p.VAR_KEYWORD for p in ps.values()):
        return ""
    names = {k for k, p in ps.items() if p.kind is not p.VAR_POSITIONAL}
    unknown = [k for k in params if k not in names]
    required = [
        k
        for k, p in ps.items()
        if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    not_required = [k for k in required if not params.get(k, {}).get("required")]
    out = []
    if unknown:
        out.append(f"parameter(s) the method does not take: {', '.join(unknown)}")
    if not_required:
        out.append(
            f"required by the method but not declared required: {', '.join(not_required)}"
        )
    return "; ".join(out)


def tier_digest(manifest_digest: str, tier: str | None, names: Iterable[str]) -> str:
    """``code_tier_digest``: the manifest version, the tier and the primitives available in it."""
    blob = json.dumps(
        [manifest_digest, tier, sorted(names)], separators=(",", ":")
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def serve_code_api(
    facade: Any, manifest: Mapping[str, Any], have: Callable[[str], bool]
) -> registry.CodeApi:
    """Self-check the server against its manifest (fail closed), then serve ``code.api``: the
    manifest digest and the primitives this run has in a tier (pi renders them from its own copy
    of the manifest and refuses a server whose digest differs)."""
    self_check(facade, manifest, have)
    api = registry.CodeApi(code_primitives(manifest, have), facade._rpc)

    def code_api(tier: str | None = None) -> dict:
        names = [p.name for p in api.primitives(tier)]
        return {
            "manifest_digest": manifest["digest"],
            "tier": tier,
            "available": names,
            "digest": tier_digest(manifest["digest"], tier, names),
        }

    facade._rpc["code.api"] = code_api
    facade._readonly_methods.add("code.api")
    facade.code_api = api
    facade.manifest = manifest
    return api
