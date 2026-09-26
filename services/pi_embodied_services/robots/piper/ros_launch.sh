#!/usr/bin/env bash
# Copyright 2026 The Show-Harness Authors (github.com/showlab/Show-Harness @137d571).
# Licensed under the Apache License, Version 2.0.
# Modified by pi-embodied: scripts/piper/{run_can.sh,run_arm.sh,run_cameras.sh,_piper_env.sh}
# merged into one launcher with a subcommand each; the adapter serials come from the
# environment (or `can --list`) instead of being hard-coded; styling dropped.
#
# Bring up the Piper ROS stack that ros_io.py talks to (AgileX cobot_magic, ROS Noetic):
#   ros_launch.sh can [--list]            both CAN buses (can_left, can_right @ 1 Mbaud), matched by the
#                                         USB-CAN adapter's stable serial: PIPER_LEFT_CAN_SERIAL /
#                                         PIPER_RIGHT_CAN_SERIAL (`can --list` shows them). Needs sudo.
#   ros_launch.sh arms [mode] [enable]    both arm nodes (piper start_ms_piper.launch; mode 1 = software
#                                         control, the mode ros_io.py commands in). Enabling CLOSES both
#                                         grippers to width 0: clear all fingers first.
#   ros_launch.sh cameras                 the Orbbec cameras (astra_camera multi_camera.launch:
#                                         /camera_f front, /camera_l /camera_r wrists)
# Run each in its own terminal, in that order; they share one roscore (started detached when none
# runs). Ctrl+C stops that launcher's nodes. Paths: COBOT_MAGIC_DIR (default ~/cobot_magic),
# CONDA_ENV (default aloha, the env with piper_sdk, for `arms`).
set -o pipefail

COBOT="${COBOT_MAGIC_DIR:-$HOME/cobot_magic}"
PIPER_WS="$COBOT/Piper_ros_private-ros-noetic/devel/setup.bash"
CAMERA_WS="$COBOT/camera_ws/devel/setup.bash"
BITRATE=1000000
ARPHRD_CAN=280

info() { echo "[piper-ros] $*"; }
die() {
	echo "[piper-ros] ERROR: $*" >&2
	exit 1
}

source_file() {
	[ -f "$1" ] || die "$2 not found at $1"
	# shellcheck disable=SC1090
	source "$1"
}

ensure_roscore() {
	timeout 3 rosnode list >/dev/null 2>&1 && return 0
	info "starting a detached roscore (log /tmp/piper_roscore.log; pkill -f roscore stops it)"
	setsid nohup roscore >/tmp/piper_roscore.log 2>&1 </dev/null &
	for _ in $(seq 1 30); do
		timeout 3 rosnode list >/dev/null 2>&1 && return 0
		sleep 1
	done
	die "roscore did not come up; see /tmp/piper_roscore.log"
}

# Stop processes matching a pattern: SIGINT, then SIGTERM, then SIGKILL. Never roscore.
kill_pattern() {
	pgrep -f "$1" >/dev/null 2>&1 || return 0
	pkill -INT -f "$1" 2>/dev/null || true
	for _ in $(seq 1 10); do
		pgrep -f "$1" >/dev/null 2>&1 || return 0
		sleep 0.3
	done
	pkill -TERM -f "$1" 2>/dev/null || true
	sleep 1
	pkill -KILL -f "$1" 2>/dev/null || true
}

# Wait (in the background) until every topic delivers one message.
watch_ready() {
	local deadline=$(($(date +%s) + $1)) what="$2" topic
	shift 2
	for topic in "$@"; do
		while ! timeout 5 rostopic echo -n1 --noarr "$topic" >/dev/null 2>&1; do
			[ "$(date +%s)" -ge "$deadline" ] && {
				info "WARNING: no message on $topic yet"
				return 1
			}
			sleep 1
		done
	done
	info "READY: $what"
}

can_ifaces() {
	local iface
	for iface in $(ls /sys/class/net/ 2>/dev/null); do
		[ "$(cat "/sys/class/net/$iface/type" 2>/dev/null)" = "$ARPHRD_CAN" ] && echo "$iface"
	done
}
serial_of() {
	local d
	d="$(readlink -f "/sys/class/net/$1/device" 2>/dev/null)"
	while [ -n "$d" ] && [ "$d" != "/" ]; do
		[ -f "$d/serial" ] && {
			cat "$d/serial"
			return 0
		}
		d="$(dirname "$d")"
	done
	return 1
}
iface_for() {
	local iface
	for iface in $(can_ifaces); do
		[ "$(serial_of "$iface")" = "$1" ] && {
			echo "$iface"
			return 0
		}
	done
	return 1
}

activate() {
	local name="$1" serial="$2" iface
	iface="$(iface_for "$serial")" || {
		info "$name: no adapter with serial $serial (ros_launch.sh can --list)"
		return 1
	}
	if [ "$iface" = "$name" ] && ip -br link show "$name" | grep -q " UP " &&
		[ "$(ip -details link show "$name" | grep -oP 'bitrate \K[0-9]+')" = "$BITRATE" ]; then
		info "$name already UP @ $BITRATE"
		return 0
	fi
	if [ "$iface" != "$name" ] && ip link show "$name" >/dev/null 2>&1; then
		sudo ip link set "$name" down 2>/dev/null || true
		sudo ip link set "$name" name "${name}_stale" 2>/dev/null || true
	fi
	sudo ip link set "$iface" down 2>/dev/null || true
	sudo ip link set "$iface" type can bitrate "$BITRATE" || return 1
	[ "$iface" = "$name" ] || sudo ip link set "$iface" name "$name" || return 1
	sudo ip link set "$name" up || return 1
	info "$name UP @ $BITRATE (was $iface)"
}

cmd_can() {
	if [ "${1:-}" = "--list" ]; then
		local iface
		for iface in $(can_ifaces); do echo "  $iface serial $(serial_of "$iface")"; done
		return 0
	fi
	: "${PIPER_LEFT_CAN_SERIAL:?set PIPER_LEFT_CAN_SERIAL (ros_launch.sh can --list)}"
	: "${PIPER_RIGHT_CAN_SERIAL:?set PIPER_RIGHT_CAN_SERIAL}"
	sudo modprobe gs_usb || die "could not load gs_usb"
	local rc=0
	activate can_left "$PIPER_LEFT_CAN_SERIAL" || rc=1
	activate can_right "$PIPER_RIGHT_CAN_SERIAL" || rc=1
	return $rc
}

cmd_arms() {
	local mode="${1:-1}" enable="${2:-true}" pattern="piper_start_ms_node" watcher=""
	source_file "$HOME/miniconda3/etc/profile.d/conda.sh" "conda"
	conda activate "${CONDA_ENV:-aloha}" || die "could not activate conda env ${CONDA_ENV:-aloha}"
	source_file /opt/ros/noetic/setup.bash "ROS Noetic"
	source_file "$PIPER_WS" "the Piper workspace"
	python3 -c "import piper_sdk" 2>/dev/null || die "piper_sdk is not importable in this env"
	ensure_roscore
	kill_pattern "$pattern"
	trap '[ -n "$watcher" ] && kill "$watcher" 2>/dev/null; kill_pattern "$pattern"' INT TERM EXIT
	info "launching both arms, mode=$mode auto_enable=$enable -- enabling CLOSES both grippers"
	watch_ready 60 "left + right joint states streaming" /puppet/joint_left /puppet/joint_right &
	watcher=$!
	roslaunch piper start_ms_piper.launch mode:="$mode" auto_enable:="$enable"
}

cmd_cameras() {
	local pattern="astra_camera" watcher=""
	source_file /opt/ros/noetic/setup.bash "ROS Noetic"
	source_file "$CAMERA_WS" "the camera workspace"
	ensure_roscore
	kill_pattern "$pattern"
	trap '[ -n "$watcher" ] && kill "$watcher" 2>/dev/null; kill_pattern "$pattern"' INT TERM EXIT
	watch_ready 90 "front + wrist cameras publishing" \
		/camera_f/color/image_raw /camera_l/color/image_raw /camera_r/color/image_raw &
	watcher=$!
	roslaunch astra_camera multi_camera.launch
}

case "${1:-}" in
can | arms | cameras)
	sub="$1"
	shift
	"cmd_$sub" "$@"
	;;
*)
	sed -n '8,19p' "$0"
	exit 2
	;;
esac
