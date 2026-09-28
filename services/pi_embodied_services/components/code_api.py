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

"""The primitive registry of an env server (``code.api``).

An env server declares its primitives once: the name the model sees, the RPC method that runs it,
its parameters, whether it moves the robot, and the API tiers it belongs to. The declaration is
what a code-as-policy caller may use and nothing else:

- ``code.api`` (read-only RPC) lists the primitives of a tier, so the agent side can render them
  into a prompt and record which API an episode ran with;
- :meth:`CodeApi.resolve` maps a call by primitive name to its RPC method, refusing undeclared
  primitives and parameters. A later ``run_code`` executor calls primitives only through it, so
  generated code reaches the same facade methods (and their safety limits) as the tools do and
  never the simulator or robot object itself.

Tiers follow CaP-X's abstraction levels: ``high`` (perception and pose-level motion), ``low``
(raw observations and small motion primitives) and ``privileged`` (ground truth, simulation only).
A primitive may carry a usage ``example``; ``code.api`` of :data:`NO_EXAMPLES` (CaP-X's S4, the
``*_reduced_api_exampleless`` configs) is the low tier with every example dropped.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

TIERS = ("high", "low", "raw", "privileged")
#: CaP-X's S4: the low tier (S3) without the primitives' usage examples.
NO_EXAMPLES = "low-noexamples"
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


@dataclass(frozen=True)
class Param:
    """One parameter of a primitive."""

    type: str
    description: str = ""
    required: bool = True
    #: An enum's values.
    values: tuple = ()
    minimum: float | None = None
    maximum: float | None = None

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
            "required": self.required,
        }
        if self.values:
            out["values"] = list(self.values)
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        return out

    def check(self, name: str, value: Any) -> None:
        """Refuse a value outside the declared enum or range (None passes: an optional's default)."""
        if value is None:
            return
        if self.values and value not in self.values:
            raise ValueError(
                f"{name} must be one of {list(self.values)}, got {value!r}"
            )
        if self.type in ("number", "integer") and isinstance(value, (int, float)):
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"{name} must be >= {self.minimum}, got {value}")
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f"{name} must be <= {self.maximum}, got {value}")


@dataclass(frozen=True)
class Primitive:
    """A primitive as the model sees it and the RPC method that runs it."""

    name: str
    method: str
    doc: str
    params: Mapping[str, Param] = field(default_factory=dict)
    mutating: bool = False
    #: One tier (the manifest's label; a tuple for the runner's bookkeeping).
    tiers: tuple[str, ...] = ("low",)

    def describe(self, examples: bool = True) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "doc": self.doc if examples else strip_examples(self.doc),
            "params": {k: p.describe() for k, p in self.params.items()},
            "mutating": self.mutating,
            "tiers": list(self.tiers),
        }


def strip_examples(doc: str) -> str:
    """Drop the ``Example:`` / ``Examples:`` sections of a Google-style docstring (CaP-X's
    exampleless tier, ``control_reduced_exampleless.py``): from the header to the next section
    header at the same indentation, or the end."""
    out: list[str] = []
    skipping: int | None = None
    for line in doc.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if skipping is not None:
            if stripped and indent <= skipping and stripped.endswith(":"):
                skipping = None
            else:
                continue
        if stripped.lower() in ("example:", "examples:"):
            skipping = indent
            continue
        out.append(line)
    return "\n".join(out).rstrip()


#: A tier with the simulator's ground truth added (``--privileged`` on a tier other than high, e.g.
#: ``low+privileged``): the tier's primitives with the privileged entries added (a privileged entry
#: of the same name replaces the tier's).
PRIVILEGED_SUFFIX = "+privileged"


def base_tier(tier: str | None) -> str | None:
    """The registry tier behind a ``code.api`` tier (S4 is the low tier)."""
    if tier is not None and tier.endswith(PRIVILEGED_SUFFIX):
        return base_tier(tier[: -len(PRIVILEGED_SUFFIX)]) + PRIVILEGED_SUFFIX
    return "low" if tier == NO_EXAMPLES else tier


class CodeApi:
    """A validated set of primitives over a facade's registered RPC methods: the whitelist every
    program call goes through (a program never reaches an RPC method outside it)."""

    def __init__(self, primitives: Iterable[Primitive], rpc: Mapping[str, Any]) -> None:
        self._all: list[Primitive] = []
        seen: set[tuple[str, bool]] = set()
        for p in primitives:
            if not p.name.isidentifier():
                raise ValueError(
                    f"primitive name {p.name!r} is not a Python identifier"
                )
            if len(p.tiers) != 1 or p.tiers[0] not in TIERS:
                raise ValueError(
                    f"primitive {p.name!r}: exactly one tier of {TIERS}, got {p.tiers}"
                )
            key = (p.name, p.tiers[0] == "privileged")
            if key in seen:
                raise ValueError(f"primitive {p.name!r} is declared twice")
            seen.add(key)
            if p.method not in rpc:
                raise ValueError(
                    f"primitive {p.name!r}: RPC method {p.method!r} is not registered"
                )
            for k, param in p.params.items():
                if not k.isidentifier() or param.type not in PARAM_TYPES:
                    raise ValueError(
                        f"primitive {p.name!r}: bad parameter {k!r} ({param.type})"
                    )
            self._all.append(p)

    def primitives(self, tier: str | None = None) -> list[Primitive]:
        """The primitives of ``tier``, in declaration order. ``privileged`` is the high tier with
        the privileged entries added (one of the same name replaces the high one); None lists
        every non-privileged primitive."""
        tier = base_tier(tier)
        if tier is not None and tier.endswith(PRIVILEGED_SUFFIX):
            base = tier[: -len(PRIVILEGED_SUFFIX)]
            if base not in ("high", "low", "raw"):
                raise ValueError(f"no privileged variant of tier {base!r}")
            priv = [p for p in self._all if p.tiers[0] == "privileged"]
            names = {p.name for p in priv}
            return [
                p for p in self._all if p.tiers[0] == base and p.name not in names
            ] + priv
        if tier is not None and tier not in TIERS:
            raise ValueError(
                f"tier must be one of {TIERS + (NO_EXAMPLES,)}, got {tier!r}"
            )
        if tier is None:
            return [p for p in self._all if p.tiers[0] != "privileged"]
        if tier == "privileged":
            priv = {p.name for p in self._all if p.tiers[0] == "privileged"}
            return [
                p
                for p in self._all
                if p.tiers[0] == "privileged"
                or (p.tiers[0] == "high" and p.name not in priv)
            ]
        return [p for p in self._all if p.tiers[0] == tier]

    def describe(self, tier: str | None = None) -> dict[str, Any]:
        """The ``code.api`` result: the tier, its primitives and a digest that names this API
        version. :data:`NO_EXAMPLES` lists the low tier without the examples."""
        examples = not (tier or "").startswith(NO_EXAMPLES)
        listed = [p.describe(examples) for p in self.primitives(tier)]
        blob = json.dumps(listed, sort_keys=True, separators=(",", ":")).encode()
        return {
            "tier": tier,
            "primitives": listed,
            "digest": hashlib.sha256(blob).hexdigest(),
        }

    def resolve(
        self, name: str, kwargs: Mapping[str, Any], tier: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """The RPC method and keyword arguments of a call to primitive ``name`` in ``tier``."""
        allowed = {p.name: p for p in self.primitives(tier)}
        p = allowed.get(name)
        if p is None:
            raise ValueError(
                f"{name!r} is not a primitive of this API (have: {', '.join(allowed) or 'none'})"
            )
        extra = [k for k in kwargs if k not in p.params]
        if extra:
            raise ValueError(f"{name}: unknown parameter(s) {', '.join(extra)}")
        missing = [
            k for k, param in p.params.items() if param.required and k not in kwargs
        ]
        if missing:
            raise ValueError(f"{name}: missing parameter(s) {', '.join(missing)}")
        for k, v in kwargs.items():
            p.params[k].check(f"{name}.{k}", v)
        return p.method, dict(kwargs)


def register_code_api(facade: Any, primitives: Iterable[Primitive]) -> CodeApi:
    """Validate ``primitives`` against ``facade``'s RPC methods and serve them as ``code.api``."""
    api = CodeApi(primitives, facade._rpc)
    facade._rpc["code.api"] = lambda tier=None: api.describe(tier)
    facade._readonly_methods.add("code.api")
    facade.code_api = api
    return api
