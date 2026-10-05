#!/usr/bin/env sh
# Launch the pi-embodied MCP server (packages/embodied/src/integrations/mcp/server.ts) for a host
# plugin. The checkout is $PI_EMBODIED_ROOT (Claude Code: the plugin's `repo` option, exported as
# CLAUDE_PLUGIN_OPTION_REPO), else the repository this plugin directory sits in (an in-tree plugin).
# The robot and its connection come from the environment (Claude Code maps the plugin options onto
# the same variables; Codex passes them through `env_vars`):
#   PI_EMBODIED_ROBOT        libero | franka | ...            (required)
#   PI_EMBODIED_ENV_URL      http://host:port[#token=HEX]      attach to a running env server, or
#   PI_EMBODIED_SERVE_ARGS   "--suite libero_10 --task 0 --seed 0"   start one (--serve -- <args>)
#   PI_EMBODIED_DEPLOYMENT   a deployment in ~/.pi/agent/embodied.json (default: `default` or the only one)
#   PI_EMBODIED_TIER         high | low | raw (default: every non-privileged tool)
#   PI_EMBODIED_PRIVILEGED   1 / true for the simulator's ground truth
#   PI_EMBODIED_CAPABILITIES a,b   what the env server was started with, when it serves no code.api
#   PI_EMBODIED_VARS         "cameras=agentview,wrist;arms="   robot variables the manifest's enums need
#   PI_EMBODIED_TIMEOUT_MS   per-call RPC timeout
#   PI_EMBODIED_NO_RESET     1: do not env.reset a simulator at start (a real arm is never reset at start: its reset tool is)
# The operator gate on a real arm (the server's own, src/integrations/mcp/gate.ts; hosts' hooks are a second
# layer and Codex 0.160 runs none): one of
#   PI_EMBODIED_CONFIRM_FILE      a path the operator writes a tool's name into before each motion (--confirm-file;
#                                 recommended; outside the model's workspace and /tmp), or
#   PI_EMBODIED_MOTION_CONFIRMED  1 exported before the host starts: the WHOLE session's motions are authorised
#                                 (the server reads it from its environment; nothing to map here).
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=${PI_EMBODIED_ROOT:-${CLAUDE_PLUGIN_OPTION_REPO:-}}
if [ -z "$root" ]; then
	candidate=$(CDPATH= cd -- "$here/../../../../.." 2>/dev/null && pwd || true)
	if [ -n "$candidate" ] && [ -f "$candidate/packages/embodied/src/integrations/mcp/server.ts" ]; then root=$candidate; fi
fi
server=${root:+$root/packages/embodied/src/integrations/mcp/server.ts}
if [ -z "$root" ] || [ ! -f "$server" ]; then
	echo "[pi-embodied-mcp] no pi-embodied checkout: set PI_EMBODIED_ROOT (or the plugin's repo option) to a built clone of LV-Robotics-Lab/pi-embodied" >&2
	exit 1
fi
if [ ! -d "$root/node_modules" ]; then
	echo "[pi-embodied-mcp] $root has no node_modules: run 'npm install && npm run build:offline' there first" >&2
	exit 1
fi
robot=${PI_EMBODIED_ROBOT:-}
if [ -z "$robot" ]; then
	echo "[pi-embodied-mcp] set PI_EMBODIED_ROBOT (or the plugin's robot option)" >&2
	exit 1
fi
set -- --robot "$robot"
[ -n "${PI_EMBODIED_DEPLOYMENT:-}" ] && set -- "$@" --deployment "$PI_EMBODIED_DEPLOYMENT"
[ -n "${PI_EMBODIED_TIER:-}" ] && set -- "$@" --tier "$PI_EMBODIED_TIER"
case "${PI_EMBODIED_PRIVILEGED:-}" in 1 | true | yes) set -- "$@" --privileged ;; esac
[ -n "${PI_EMBODIED_CAPABILITIES:-}" ] && set -- "$@" --capabilities "$PI_EMBODIED_CAPABILITIES"
[ -n "${PI_EMBODIED_TIMEOUT_MS:-}" ] && set -- "$@" --timeout "$PI_EMBODIED_TIMEOUT_MS"
case "${PI_EMBODIED_NO_RESET:-}" in 1 | true | yes) set -- "$@" --no-reset ;; esac
[ -n "${PI_EMBODIED_CONFIRM_FILE:-}" ] && set -- "$@" --confirm-file "$PI_EMBODIED_CONFIRM_FILE"
if [ -n "${PI_EMBODIED_VARS:-}" ]; then
	old_ifs=$IFS
	IFS=';'
	for v in $PI_EMBODIED_VARS; do
		IFS=$old_ifs
		[ -n "$v" ] && set -- "$@" --var "$v"
		IFS=';'
	done
	IFS=$old_ifs
fi
if [ -n "${PI_EMBODIED_ENV_URL:-}" ]; then
	set -- "$@" --env "$PI_EMBODIED_ENV_URL"
elif [ -n "${PI_EMBODIED_SERVE_ARGS:-}" ]; then
	# Word-split on purpose: the env server's own argument list.
	# shellcheck disable=SC2086
	set -- "$@" --serve -- $PI_EMBODIED_SERVE_ARGS
else
	echo "[pi-embodied-mcp] set PI_EMBODIED_ENV_URL (a running env server) or PI_EMBODIED_SERVE_ARGS (start one)" >&2
	exit 1
fi
exec node --experimental-strip-types "$server" "$@"
