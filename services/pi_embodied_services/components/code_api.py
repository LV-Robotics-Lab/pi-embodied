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
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

TIERS = ("high", "low", "privileged")
PARAM_TYPES = ("number", "integer", "boolean", "string", "vec3", "array", "object")


@dataclass(frozen=True)
class Param:
    """One parameter of a primitive."""

    type: str
    description: str = ""
    required: bool = True

    def describe(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "description": self.description,
            "required": self.required,
        }


@dataclass(frozen=True)
class Primitive:
    """A primitive as the model sees it and the RPC method that runs it."""

    name: str
    method: str
    doc: str
    params: Mapping[str, Param] = field(default_factory=dict)
    mutating: bool = False
    tiers: tuple[str, ...] = ("high", "low")

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "doc": self.doc,
            "params": {k: p.describe() for k, p in self.params.items()},
            "mutating": self.mutating,
            "tiers": list(self.tiers),
        }


class CodeApi:
    """A validated set of primitives over a facade's registered RPC methods."""

    def __init__(self, primitives: Iterable[Primitive], rpc: Mapping[str, Any]) -> None:
        self._by_name: dict[str, Primitive] = {}
        for p in primitives:
            if not p.name.isidentifier():
                raise ValueError(
                    f"primitive name {p.name!r} is not a Python identifier"
                )
            if p.name in self._by_name:
                raise ValueError(f"primitive {p.name!r} is declared twice")
            if p.method not in rpc:
                raise ValueError(
                    f"primitive {p.name!r}: RPC method {p.method!r} is not registered"
                )
            unknown = [t for t in p.tiers if t not in TIERS]
            if not p.tiers or unknown:
                raise ValueError(
                    f"primitive {p.name!r}: tiers must be among {TIERS}, got {p.tiers}"
                )
            if "privileged" in p.tiers and len(p.tiers) > 1:
                raise ValueError(
                    f"primitive {p.name!r}: a privileged primitive is in no other tier"
                )
            for k, param in p.params.items():
                if not k.isidentifier() or param.type not in PARAM_TYPES:
                    raise ValueError(
                        f"primitive {p.name!r}: bad parameter {k!r} ({param.type})"
                    )
            self._by_name[p.name] = p

    def primitives(self, tier: str | None = None) -> list[Primitive]:
        """The primitives of ``tier`` (all but privileged when None), in declaration order."""
        if tier is not None and tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
        if tier is None:
            return [p for p in self._by_name.values() if "privileged" not in p.tiers]
        if tier == "privileged":
            # The privileged tier adds ground truth to the high-level API.
            return [
                p
                for p in self._by_name.values()
                if "high" in p.tiers or "privileged" in p.tiers
            ]
        return [p for p in self._by_name.values() if tier in p.tiers]

    def describe(self, tier: str | None = None) -> dict[str, Any]:
        """The ``code.api`` result: the tier, its primitives and a digest that names this API version."""
        listed = [p.describe() for p in self.primitives(tier)]
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
        return p.method, dict(kwargs)


def register_code_api(facade: Any, primitives: Iterable[Primitive]) -> CodeApi:
    """Validate ``primitives`` against ``facade``'s RPC methods and serve them as ``code.api``."""
    api = CodeApi(primitives, facade._rpc)
    facade._rpc["code.api"] = lambda tier=None: api.describe(tier)
    facade._readonly_methods.add("code.api")
    facade.code_api = api
    return api
