/**
 * Deployment config (PARAMS.md 3): where services listen, which python runs an env server, where
 * outputs land. These are facts about the machine, not experiment parameters, so they are not flags
 * and are not recorded in `params` (result.json records a hash of the deployment for provenance).
 *
 * Files: `~/.pi/agent/embodied.json` (global; `$PI_EMBODIED_CONFIG` replaces its path) and
 * `<cwd>/.pi/embodied.json` (project; wins key by key). `--deployment <name>` picks one of
 * `deployments` (default: `default`, else the only one). Built-in defaults and the environment
 * (`PI_EMBODIED_SERVICES`, `PI_EMBODIED_PYTHON`, ...) sit below the file:
 * built-in < environment < deployment for services and python; per-worker facts the eval scripts set
 * (`PI_EMBODIED_CUDA_DEVICE`, `PI_EMBODIED_DIRS_<KIND>`) win over the deployment.
 *
 * Readers: `service(pi, name)`, `python(pi, venv)`, `dir(pi, kind)`, `servicesDir(pi)`,
 * `ffmpeg(pi)`, `cudaDevice(pi)`. `configProblem(pi)` names a malformed file or an unknown
 * deployment (robot.ts refuses to start on it); `/embodied-config` prints the effective config.
 */

import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export type Deployment = {
	services?: Record<string, string>;
	python?: Record<string, string>;
	services_dir?: string;
	dirs?: Record<string, string>;
	ffmpeg?: string;
	cuda_device?: string;
	/** Colon-separated setup.bash files sourced before a ROS env server starts (Piper); empty: pi's environment. */
	ros_setup?: string;
	/** Per-role auxiliary VLMs (provider/id) over --aux-model: the rare role that needs another model. */
	aux?: Partial<Record<AuxRole, string>>;
};
export type ConfigFile = { deployments?: Record<string, Deployment>; presets?: Record<string, Record<string, string>> };

/** Built-in service endpoints: the ports serve.sh starts them on. Services without one must be configured. */
export const SERVICE_DEFAULTS: Record<string, string> = {
	sam3: "http://127.0.0.1:18300",
	molmo: "http://127.0.0.1:18400",
	vla: "http://127.0.0.1:18200",
	rldx: "http://127.0.0.1:18500",
	lingbot: "ws://127.0.0.1:18400",
	finetuned: "http://127.0.0.1:8010/v1",
};

/** Every service key a robot may read; a deployment naming another key is refused (a typo would be silent). */
export const SERVICE_KEYS = [
	"sam3",
	"molmo",
	"vla",
	"openvla",
	"openvla_oft",
	"gr00t",
	"ik",
	"contact_graspnet",
	"graspgenx",
	"anygrasp",
	"graspnet1b",
	"anyplace",
	"unidepth",
	"rldx",
	"lingbot",
	"finetuned",
] as const;
export type ServiceKey = (typeof SERVICE_KEYS)[number];

/** The auxiliary VLM roles: VDM, the units verifier and video_ref, check_attached (and suggest_grasp). */
export const AUX_ROLES = ["vdm", "verify", "attach"] as const;
export type AuxRole = (typeof AUX_ROLES)[number];

export const DIR_KINDS = [
	"artifacts",
	"memory",
	"memory_out",
	"logs",
	"video",
	"flywheel",
	"flash_plans",
	"api_slots",
] as const;
export type DirKind = (typeof DIR_KINDS)[number];

const BUILTIN_SERVICES_DIR = fileURLToPath(new URL("../../../../services", import.meta.url));

export const globalConfigPath = () =>
	process.env.PI_EMBODIED_CONFIG ||
	join(process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi", "agent"), "embodied.json");
export const projectConfigPath = (cwd = process.cwd()) => join(cwd, ".pi", "embodied.json");

type Loaded = { file: ConfigFile; problem?: string };

function readOne(path: string): Loaded {
	if (!existsSync(path)) return { file: {} };
	try {
		const j = JSON.parse(readFileSync(path, "utf8"));
		if (!j || typeof j !== "object" || Array.isArray(j)) return { file: {}, problem: `${path}: not a JSON object` };
		return { file: j as ConfigFile };
	} catch (e) {
		return { file: {}, problem: `${path}: ${(e as Error).message}` };
	}
}

const merge = <T extends object>(a: T | undefined, b: T | undefined): T => {
	const out: Record<string, unknown> = { ...(a ?? {}) };
	for (const [k, v] of Object.entries(b ?? {})) {
		const prev = out[k];
		out[k] =
			v && typeof v === "object" && !Array.isArray(v) && prev && typeof prev === "object"
				? merge(prev as object, v)
				: v;
	}
	return out as T;
};

/** Both files, project over global (read on every call: sync, small, and a test may swap them). */
export function loadConfig(cwd = process.cwd()): Loaded {
	const g = readOne(globalConfigPath());
	const p = readOne(projectConfigPath(cwd));
	return { file: merge(g.file, p.file), problem: g.problem ?? p.problem };
}

const name = (pi: ExtensionAPI) => String(pi.getFlag("deployment") ?? "").trim();

/** The selected deployment's name, or a problem (unknown name, ambiguous default, malformed file). */
function select(pi: ExtensionAPI): { name: string; deployment: Deployment; problem?: string } {
	const { file, problem } = loadConfig();
	const all = file.deployments ?? {};
	if (problem) return { name: "", deployment: {}, problem };
	const asked = name(pi);
	if (asked) {
		if (!all[asked])
			return {
				name: asked,
				deployment: {},
				problem: `--deployment ${asked} is not in ${globalConfigPath()} or ${projectConfigPath()} (known: ${Object.keys(all).join(", ") || "none"})`,
			};
		return { name: asked, deployment: all[asked] };
	}
	if (all.default) return { name: "default", deployment: all.default };
	const names = Object.keys(all);
	if (names.length === 1) return { name: names[0], deployment: all[names[0]] };
	return { name: "", deployment: {} };
}

/** The deployment in effect (empty when no file names one). */
export const deployment = (pi: ExtensionAPI) => select(pi).deployment;

/** Why the config cannot be used (robot.ts refuses to start), else undefined. */
export function configProblem(pi: ExtensionAPI): string | undefined {
	const s = select(pi);
	if (s.problem) return s.problem;
	const bad = Object.keys(s.deployment.services ?? {}).filter((k) => !(SERVICE_KEYS as readonly string[]).includes(k));
	if (bad.length)
		return `deployment ${s.name}: unknown services.${bad.join(", services.")} (known: ${SERVICE_KEYS.join(", ")})`;
	const badDir = Object.keys(s.deployment.dirs ?? {}).filter((k) => !(DIR_KINDS as readonly string[]).includes(k));
	if (badDir.length)
		return `deployment ${s.name}: unknown dirs.${badDir.join(", dirs.")} (known: ${DIR_KINDS.join(", ")})`;
	const badAux = Object.keys(s.deployment.aux ?? {}).filter((k) => !(AUX_ROLES as readonly string[]).includes(k));
	if (badAux.length)
		return `deployment ${s.name}: unknown aux.${badAux.join(", aux.")} (known: ${AUX_ROLES.join(", ")})`;
	return undefined;
}

/**
 * The model (provider/id) of an auxiliary VLM role: the deployment's `aux.<role>`, else `--aux-model`,
 * else "" (the session's model). A result records every role's (`auxModels`).
 */
export function auxModel(pi: ExtensionAPI, role: AuxRole): string {
	return deployment(pi).aux?.[role]?.trim() || String(pi.getFlag("aux-model") ?? "").trim();
}

/** For result.json: each role's model, null for the session's. */
export const auxModels = (pi: ExtensionAPI) =>
	Object.fromEntries(AUX_ROLES.map((r) => [r, auxModel(pi, r) || null])) as Record<AuxRole, string | null>;

/**
 * A service's endpoint: the deployment's, else the built-in port, else "" (not configured). A
 * deployment switches a built-in service off with `"off"` (e.g. no Pi0.5 on this machine): "" too.
 */
export function service(pi: ExtensionAPI, key: ServiceKey): string {
	const url = String(deployment(pi).services?.[key] ?? SERVICE_DEFAULTS[key] ?? "").trim();
	return url === "off" ? "" : url;
}

/** A service a switch turned on: its endpoint, or the reason it is missing. */
export function requireService(pi: ExtensionAPI, key: ServiceKey, by: string): { url: string } | { error: string } {
	const url = service(pi, key);
	return url ? { url } : { error: `${by} needs services.${key} in the deployment config (${globalConfigPath()})` };
}

/**
 * The python of a venv (a robot's name, `flywheel`, ...): `python.<venv>`, else the venv's own
 * environment variables (`env`, e.g. PI_EMBODIED_FLYWHEEL_PYTHON: a venv-specific setting beats the
 * generic one), else `python.default`, else PI_EMBODIED_PYTHON.
 */
export function python(pi: ExtensionAPI, venv: string, env: string[] = []): string {
	const p = deployment(pi).python ?? {};
	for (const v of [p[venv], ...env.map((e) => process.env[e]), p.default, process.env.PI_EMBODIED_PYTHON])
		if (v?.trim()) return v.trim();
	return "python";
}

/** The pi-embodied services tree the env servers run from. */
export const servicesDir = (pi: ExtensionAPI) =>
	deployment(pi).services_dir?.trim() || process.env.PI_EMBODIED_SERVICES || BUILTIN_SERVICES_DIR;

/**
 * An output/input directory; `fallback` is the robot's default (e.g. under the session dir).
 * `$PI_EMBODIED_DIRS_<KIND>` (e.g. PI_EMBODIED_DIRS_LOGS) wins over the deployment: the eval
 * scripts set it per worker or per episode.
 */
export const dir = (pi: ExtensionAPI, kind: DirKind, fallback = "") =>
	process.env[`PI_EMBODIED_DIRS_${kind.toUpperCase()}`]?.trim() || deployment(pi).dirs?.[kind]?.trim() || fallback;

export const ffmpeg = (pi: ExtensionAPI) => deployment(pi).ffmpeg?.trim() || "";

/** The ROS setup files a ROS env server sources first (`ros_setup`, colon-separated). */
export const rosSetup = (pi: ExtensionAPI) =>
	(deployment(pi).ros_setup ?? "")
		.split(":")
		.map((s) => s.trim())
		.filter(Boolean);

/** The GPU ordinal an env server renders on: the environment (eval-parallel.sh's per-worker GPU) over the deployment. */
export const cudaDevice = (pi: ExtensionAPI) =>
	process.env.PI_EMBODIED_CUDA_DEVICE?.trim() || deployment(pi).cuda_device?.trim() || "";

/** For result.json: the deployment's name and a hash of its content (provenance, not a parameter). */
export function deploymentRecord(pi: ExtensionAPI): { deployment: string; deployment_sha: string | null } {
	const s = select(pi);
	const body = JSON.stringify(s.deployment);
	return {
		deployment: s.name,
		deployment_sha: s.name ? createHash("sha256").update(body).digest("hex").slice(0, 16) : null,
	};
}

/** Register `--deployment` and `/embodied-config` (robot.ts, once per robot). */
export function registerConfig(pi: ExtensionAPI) {
	pi.registerFlag("deployment", {
		type: "string",
		default: "",
		description:
			"Deployment in ~/.pi/agent/embodied.json or .pi/embodied.json (where services, python and outputs live)",
	});
	pi.registerCommand?.("embodied-config", {
		description: "Print the effective deployment config",
		handler: async (_args, ctx) => {
			const s = select(pi);
			const text = JSON.stringify(
				{
					files: [globalConfigPath(), projectConfigPath()],
					deployment: s.name || null,
					problem: s.problem ?? configProblem(pi) ?? null,
					services: Object.fromEntries(SERVICE_KEYS.map((k) => [k, service(pi, k) || null])),
					services_dir: servicesDir(pi),
					python: { default: python(pi, "default"), ...(s.deployment.python ?? {}) },
					dirs: s.deployment.dirs ?? {},
					ffmpeg: ffmpeg(pi) || null,
					cuda_device: cudaDevice(pi) || null,
				},
				null,
				2,
			);
			ctx.ui.notify(text, "info");
		},
	});
}
