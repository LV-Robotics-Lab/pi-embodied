/**
 * Tests' deployment config (../../src/infra/config.ts): `useDeployment` writes a config file with
 * one `default` deployment and points `$PI_EMBODIED_CONFIG` at it (the project file is left alone:
 * tests run from the package dir, which has none). `splitFlags` takes a test's flag object written
 * the short way (`{ sam3: url, python: PY, ik: url }`) and returns the pi flags (switches on) and the
 * deployment those URLs and paths belong to, so a fake pi keeps one argument.
 */

import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Deployment } from "../../src/infra/config.ts";

const GRASP = {
	"contact-graspnet": "contact_graspnet",
	graspgenx: "graspgenx",
	anygrasp: "anygrasp",
	graspnet1b: "graspnet1b",
};
const ADAPTERS = { openvla: "openvla", "openvla-oft": "openvla_oft", gr00t: "gr00t" };
const SERVICES = ["sam3", "molmo", "vla", "rldx", "lingbot"];
const DIRS: Record<string, string> = {
	out: "artifacts",
	"memory-dir": "memory",
	"output-dir": "memory_out",
	"log-dir": "logs",
	"video-dir": "video",
	"flywheel-root": "flywheel",
	"flash-plans": "flash_plans",
	"api-slots": "api_slots",
	"serve-log-dir": "logs",
};

let file: string | undefined;

/** Make `d` the deployment in effect (until the next call); returns the config file. */
export function useDeployment(d: Deployment): string {
	file ??= join(mkdtempSync(join(tmpdir(), "pi-embodied-config-")), "embodied.json");
	writeFileSync(file, JSON.stringify({ deployments: { default: d } }));
	process.env.PI_EMBODIED_CONFIG = file;
	return file;
}

/** Split a test's flags into pi flags and a deployment (see the module comment). */
export function splitFlags(all: Record<string, unknown>): { flags: Record<string, unknown>; deployment: Deployment } {
	const flags: Record<string, unknown> = {};
	const d: Required<Pick<Deployment, "services" | "python" | "dirs">> & Deployment = {
		services: {},
		python: {},
		dirs: {},
	};
	const grasp: string[] = [];
	const adapters: string[] = [];
	/** A service switched on by a URL-valued test flag: record the URL, set the switch. */
	const on = (key: string, url: string, flag: string, value: unknown) => {
		if (!url.trim()) return;
		d.services[key] = url;
		flags[flag] = value;
	};
	for (const [k, v] of Object.entries(all)) {
		const s = typeof v === "string" ? v : "";
		if (k === "services") d.services_dir = s;
		else if (k === "python") d.python.default = s;
		else if (k === "robocasa-python") d.python.robocasa = s;
		else if (k === "flywheel-python") d.python.flywheel = s;
		else if (k === "viser-python") d.python.viser = s;
		else if (k === "xpolicy-python") d.python.xpolicy = s;
		else if (k === "ffmpeg") d.ffmpeg = s;
		else if (k === "cuda-device" || k === "gpu-id") d.cuda_device = s;
		else if (k in DIRS) d.dirs[DIRS[k]] = s;
		else if (k === "ft-endpoint") d.services.finetuned = s;
		else if (k === "molmo" && s === "off") on("molmo", s, "flash-reanchor", "off");
		else if (SERVICES.includes(k)) d.services[k] = s;
		else if (k === "robot-sam3") on("sam3", s, "segment", true);
		else if (k === "robot-vla") on("vla", s, "vla", true);
		else if (k === "ik" && typeof v === "string") on("ik", s, "ik", true);
		else if (k === "unidepth" || k === "robot-unidepth") on("unidepth", s, "depth", "unidepth");
		else if (k === "anyplace") on("anyplace", s, "place", "anyplace");
		else if (k in GRASP) {
			const key = GRASP[k as keyof typeof GRASP];
			if (s) grasp.push(key);
			if (s) d.services[key] = s;
		} else if (k in ADAPTERS) {
			const key = ADAPTERS[k as keyof typeof ADAPTERS];
			if (s) adapters.push(k);
			if (s) d.services[key] = s;
		} else flags[k] = v;
	}
	if (grasp.length) flags.grasp = grasp.join(",");
	if (adapters.length) flags["vla-adapter"] = adapters.join(",");
	return { flags, deployment: d };
}

/**
 * `splitFlags`, then `useDeployment` with its deployment; rewrites `all` in place (a test may add
 * flags to the same object later, as pi sets CLI values after load) and returns it.
 */
export function deployFlags<T extends Record<string, unknown>>(all: T): T {
	const { flags, deployment } = splitFlags(all);
	useDeployment(deployment);
	for (const k of Object.keys(all)) delete all[k];
	Object.assign(all, flags);
	return all;
}
