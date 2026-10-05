#!/usr/bin/env sh
# PreToolUse hook: the operator gate for pi-embodied motion tools, run by Claude Code (--decision ask)
# and Codex (--decision deny). Finds the checkout like pi-embodied-mcp.sh and runs
# packages/embodied/src/integrations/mcp/hook.ts on the host's stdin JSON; the hook prints a
# decision for high-risk motions and nothing for the rest. When it cannot run, exit 2 blocks the call.
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=${PI_EMBODIED_ROOT:-${CLAUDE_PLUGIN_OPTION_REPO:-}}
if [ -z "$root" ]; then
	candidate=$(CDPATH= cd -- "$here/../../../../.." 2>/dev/null && pwd || true)
	if [ -n "$candidate" ] && [ -f "$candidate/packages/embodied/src/integrations/mcp/hook.ts" ]; then root=$candidate; fi
fi
hook=${root:+$root/packages/embodied/src/integrations/mcp/hook.ts}
if [ -z "$root" ] || [ ! -f "$hook" ]; then
	echo "pi-embodied motion gate: no pi-embodied checkout (set PI_EMBODIED_ROOT); refusing the robot call" >&2
	exit 2
fi
exec node --experimental-strip-types "$hook" "$@"
