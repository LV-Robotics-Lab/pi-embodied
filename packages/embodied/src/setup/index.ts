/**
 * Onboarding: the package's only default extension. It points a new user to `/embodied-setup` once
 * per project, and `/embodied-setup` asks for the robot, the mode, where the services run and the
 * planner, writes the experiment directory's `.pi/settings.json`, and hands the install to the agent
 * (the embodied-quickstart skill), which runs services/setup.sh and the preflight with its own tools
 * once the user has confirmed the plan (pi does not gate each tool call).
 *
 * A robot extension replaces the coding tools, so robots never load by default: the experiment
 * directory's settings load one robot (and the dashboard) through `extensions`, and a delta entry
 * for this package (`autoload: false`, `-src/setup/index.ts`) drops this extension there. Package
 * filters only narrow what the manifest declares (docs/packages.md), which is why the robots are
 * listed as extension paths and not as a filter of this package.
 *
 * The notice is shown once per project: a `pi-embodied-setup-notice` session entry records it, and
 * any session of the project (the session directory) that holds one suppresses it.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@earendil-works/pi-coding-agent";

/** The package root and the services checkout next to it (absent in an npm install). */
export const PKG = fileURLToPath(new URL("../..", import.meta.url)).replace(/\/$/, "");
const SERVICES = join(PKG, "../../services");
const REPO = "https://github.com/LV-Robotics-Lab/pi-embodied.git";
const SKILL = join(PKG, "skills/embodied-quickstart/SKILL.md");
export const NOTICE_ENTRY = "pi-embodied-setup-notice";
export const COMMAND = "embodied-setup";
export const NOTICE =
	"pi-embodied: run /embodied-setup to pick a robot, a mode and where the services run; the agent then installs and checks them.";

type Mode = "tools" | "units" | "finetuned" | "flash";
export type Robot = {
	/** services/setup.sh target. */
	id: string;
	label: string;
	/** The robot extension, relative to the package. */
	extension: string;
	needs: string;
	modes: Mode[];
	/** Example task flags for the launch command. */
	task: string;
	/** What `setup.sh --weights` fetches (tools mode). */
	weights?: string;
	/** Fine-tuned mode: model and flags. */
	finetuned?: string;
	/** Fine-tuned mode: the released adapter `setup.sh finetuned` fetches (FT_ADAPTER); unset when the user serves their own. */
	adapter?: string;
};

const SIM_FT = "--model finetuned/qwen3_5_2b_showharness_sim";
export const ROBOTS: Robot[] = [
	{
		id: "libero-pro",
		label: "LIBERO-PRO (sim)",
		extension: "src/libero/index.ts",
		needs: "NVIDIA GPU (MuJoCo EGL)",
		modes: ["tools", "units", "finetuned", "flash"],
		task: "--libero-type pro --suite libero_10 --task 0 --seed 0",
		weights: "Pi0.5 LIBERO SFT + SAM3 (gated: HF_TOKEN)",
		finetuned: "--model finetuned/local --ft-model <served adapter> --ft-prompt v5",
	},
	{
		id: "libero",
		label: "LIBERO (sim)",
		extension: "src/libero/index.ts",
		needs: "NVIDIA GPU (MuJoCo EGL)",
		modes: ["tools", "units", "finetuned", "flash"],
		task: "--libero-type standard --suite libero_10 --task 0 --seed 0",
		weights: "Pi0.5 LIBERO SFT + SAM3 (gated: HF_TOKEN)",
		finetuned: "--model finetuned/local --ft-model <served adapter> --ft-prompt v5",
	},
	{
		id: "maniskill",
		label: "ManiSkill 3 (sim)",
		extension: "src/maniskill/index.ts",
		needs: "NVIDIA GPU (Vulkan)",
		modes: ["tools", "units", "finetuned"],
		task: "--env-id PickCube-v1 --seed 0",
		finetuned: SIM_FT,
		adapter: "qwen3_5_2b_sim",
	},
	{
		id: "metaworld",
		label: "Metaworld MT50 (sim)",
		extension: "src/metaworld/index.ts",
		needs: "NVIDIA GPU (MuJoCo EGL)",
		modes: ["tools", "units"],
		task: "--task reach-v3 --seed 0",
	},
	{
		id: "robosuite",
		label: "Robosuite (sim, CaP-X tasks)",
		extension: "src/robosuite/index.ts",
		needs: "NVIDIA GPU (MuJoCo EGL), SAM3 for segment",
		modes: ["tools", "units"],
		task: "--task Lift --seed 0",
	},
	{
		id: "robocasa",
		label: "RoboCasa365 (sim)",
		extension: "src/robocasa/index.ts",
		needs: "NVIDIA GPU, ~10 GB kitchen assets",
		modes: ["tools"],
		task: "--task-name OpenDrawer --split target --seed 0",
		weights: "RLDX-1-FT-RC365 VLA",
	},
	{
		id: "robotwin",
		label: "RoboTwin 2 (sim, two arms)",
		extension: "src/robotwin/index.ts",
		needs: "NVIDIA GPU (SAPIEN/Vulkan), cuRobo build",
		modes: ["tools"],
		task: "--task-name beat_block_hammer --task-config demo_randomized --seed 0",
		weights: "LingBot-VLA RoboTwin EEF",
	},
	{
		id: "robolab",
		label: "RoboLab (Isaac Sim 6.1)",
		extension: "src/robolab/index.ts",
		needs: "RTX GPU, Isaac Sim 6.1 / Isaac Lab 3.0",
		modes: ["tools", "units", "finetuned"],
		task: "--task BananaInBowlTask --seed 0",
		finetuned: SIM_FT,
		adapter: "qwen3_5_2b_sim",
	},
	{
		id: "franka",
		label: "Franka (real, RLinf)",
		extension: "src/franka/index.ts",
		needs: "real Franka, RLinf controller + Ray, an operator at the e-stop",
		modes: ["tools", "units"],
		task: "--robot-config <yaml> --task 0 --operator=true",
	},
	{
		id: "franka-polymetis",
		label: "Franka (real, Polymetis NUC)",
		extension: "src/franka/index.ts",
		needs: "real Franka on a Polymetis NUC, an operator at the e-stop",
		modes: ["tools", "units"],
		task: "--robot-backend polymetis --robot-config <yaml> --operator=true",
	},
	{
		id: "dual-franka",
		label: "Dual Franka (real)",
		extension: "src/dual_franka/index.ts",
		needs: "two real Frankas, RLinf + Ray, an operator at the e-stop",
		modes: ["tools", "units"],
		task: "--robot-config <yaml> --operator=true",
	},
	{
		id: "piper",
		label: "AgileX Piper (real)",
		extension: "src/piper/index.ts",
		needs: "Piper arm, ROS Noetic, an operator at the e-stop",
		modes: ["tools", "units", "finetuned"],
		task: "--robot-config <yaml> --task banana_plate --operator=true",
		finetuned: "--model finetuned/qwen3_5_2b_showharness_ft --ft-prompt v4-piper",
		adapter: "qwen3_5_2b",
	},
];

const MODES: Record<Mode, string> = {
	tools: "tools: the planner calls the robot's own tools (VLA, segmentation, motion)",
	units: "units: Show-Harness action units (act: move, rotate, grasp, release); no VLA needed",
	finetuned: "fine-tuned: a small VLM + LoRA picks each unit (served by vLLM)",
	flash: "flash: replay a recorded plan with live grounding, no LLM",
};

type Settings = Record<string, unknown> & { packages?: unknown[]; extensions?: unknown[] };
export type Choice = {
	robot: Robot;
	mode: Mode;
	remote?: { host: string; checkout: string };
	services: string;
	weights: boolean;
	model?: string;
	dir: string;
};

/** A robot extension is active when `/robot-task` (registered by every robot's base) exists. */
const robotActive = (pi: ExtensionAPI) => pi.getCommands().some((c) => c.name === "robot-task");

function noticeSeen(ctx: ExtensionContext): boolean {
	if (ctx.sessionManager.getEntries().some((e) => e.type === "custom" && e.customType === NOTICE_ENTRY)) return true;
	const mark = `"customType":"${NOTICE_ENTRY}"`;
	try {
		const dir = ctx.sessionManager.getSessionDir();
		return readdirSync(dir).some((f) => f.endsWith(".jsonl") && readFileSync(join(dir, f), "utf8").includes(mark));
	} catch {
		return false;
	}
}

/** How this package is declared (npm/git as configured, else its absolute path), for the delta entry. */
function packageSource(pi: ExtensionAPI): string {
	const info = pi.getCommands().find((c) => c.name === COMMAND)?.sourceInfo;
	if (info?.origin === "package" && /^(npm:|git:|https?:|ssh:|git@)/.test(info.source)) return info.source;
	return PKG;
}

/** The launch flags for a choice (after `pi`). */
export function launchFlags(c: Choice): string {
	const mode =
		c.mode === "units"
			? "--units=true"
			: c.mode === "finetuned"
				? `--units=true ${c.robot.finetuned ?? ""}`
				: c.mode === "flash"
					? "--model flash/replay"
					: "";
	return [c.robot.task, mode, "--dashboard=true"].filter(Boolean).join(" ");
}

/** The experiment directory's settings: this package without onboarding, the robot, the dashboard, the planner. */
export function experimentSettings(c: Choice, source: string, root: string, existing: Settings = {}): Settings {
	const src = (p: string) => `${root}/${p}`;
	const extensions = [src(c.robot.extension), src("src/dashboard/index.ts")];
	if (c.mode === "finetuned") extensions.push(src("src/finetuned/index.ts"));
	const same = (p: unknown) => {
		const s = typeof p === "string" ? p : (p as { source?: unknown })?.source;
		return s === source || s === root;
	};
	const out: Settings = {
		...existing,
		packages: [
			...(existing.packages ?? []).filter((p) => !same(p)),
			{ source, autoload: false, extensions: ["-src/setup/index.ts"] },
		],
		extensions: [
			...(existing.extensions ?? []).filter((e) => typeof e !== "string" || !e.startsWith(`${root}/src/`)),
			...extensions,
		],
	};
	if (c.model) {
		const i = c.model.indexOf("/");
		out.defaultProvider = c.model.slice(0, i);
		out.defaultModel = c.model.slice(i + 1);
	}
	return out;
}

function handoff(c: Choice, settingsText: string): string {
	const setup = `${c.services}/setup.sh ${c.robot.id}${c.weights ? " --weights" : ""}`;
	const venv = `${c.services}/.venv-${c.robot.id}`;
	const where = c.remote
		? `remote: run everything on \`ssh ${c.remote.host}\` in the checkout ${c.remote.checkout} (clone ${REPO} there if missing); the robot runs where pi runs, so pi and the experiment directory go on the box too`
		: existsSync(c.services)
			? `local, services at ${c.services}`
			: `local; no services checkout next to the package: clone ${REPO} to ${dirname(c.services)}`;
	const settings = c.remote
		? `write this to <experiment dir on the box>/.pi/settings.json (the package paths become ${c.remote.checkout}/packages/embodied):\n${settingsText}`
		: `${join(c.dir, ".pi/settings.json")} is written`;
	// Fine-tuned mode: the adapter and its base model, and the vLLM server the launch's --ft-endpoint (default :8010) expects.
	const ft = c.robot.adapter ? `FT_ADAPTER=${c.robot.adapter} ${c.services}/setup.sh finetuned` : undefined;
	const finetuned =
		c.mode === "finetuned"
			? [
					ft
						? `- adapter: \`${ft}\` (finetuned/download.py fetches the adapter and its base model, pinned and verified)`
						: "- adapter: none is released for this robot; ask me which adapter to serve and where its files are",
					`- serve: \`MODEL=<base dir> LORA=<name>=<adapter dir> VLLM_VENV=<venv with vllm> bash ${c.services}/pi_embodied_services/finetuned/serve.sh\` (${ft ? "setup.sh finetuned prints the exact command; " : ""}the launch's --ft-endpoint defaults to http://127.0.0.1:8010/v1); it must be running for the episodes`,
				]
			: [];
	return [
		`/skill:embodied-quickstart Install and verify pi-embodied with the choices I confirmed in /embodied-setup (skill file: ${SKILL}):`,
		`- robot: ${c.robot.label}; setup target \`${c.robot.id}\`; needs ${c.robot.needs}`,
		`- mode: ${MODES[c.mode]}`,
		`- services: ${where}`,
		`- install: \`${setup}\`${c.weights ? ` (weights: ${c.robot.weights})` : ""}`,
		c.model ? `- planner: ${c.model}` : "- planner: the policy named in the launch flags",
		...finetuned,
		`- experiment dir: ${c.dir}; ${settings}`,
		`- launch: \`cd ${c.dir} && source ${venv}/pi-embodied.env && pi ${launchFlags(c)}\``,
		"- trust: the first pi in the experiment dir asks whether to trust the project (its .pi/settings.json loads the robot extension); I answer yes there, or the robot does not load (/trust saves the decision)",
		"Run the install, then the preflight, fix what fails, and end with the launch command. Ask me before sudo, anything touching real hardware, or downloads not listed here.",
	].join("\n");
}

async function setup(pi: ExtensionAPI, ctx: ExtensionCommandContext): Promise<void> {
	if (robotActive(pi)) {
		ctx.ui.notify("A robot is already loaded here; run /embodied-setup from a directory without one.", "warning");
		return;
	}
	if (!ctx.hasUI) {
		ctx.ui.notify("/embodied-setup needs an interactive session", "error");
		return;
	}
	const ui = ctx.ui;
	const robotOpts = ROBOTS.map((r) => `${r.label}: ${r.needs}`);
	const robot = ROBOTS[robotOpts.indexOf((await ui.select("Robot or benchmark", robotOpts)) ?? "")];
	if (!robot) return;
	const modeOpts = robot.modes.map((m) => MODES[m]);
	const mode = robot.modes[modeOpts.indexOf((await ui.select("Mode", modeOpts)) ?? "")];
	if (!mode) return;

	const where = await ui.select("Where do the services run?", ["here (this machine)", "a remote GPU box over ssh"]);
	if (!where) return;
	let remote: Choice["remote"];
	let services = SERVICES;
	if (where.startsWith("a remote")) {
		const host = (await ui.input("ssh host", "user@gpu-box"))?.trim();
		if (!host) return;
		const checkout = (await ui.input("pi-embodied checkout on the box", "~/pi-embodied"))?.trim() || "~/pi-embodied";
		remote = { host, checkout };
		services = `${checkout}/services`;
	} else if (!existsSync(SERVICES)) {
		const checkout = (
			await ui.input(`pi-embodied checkout (cloned from ${REPO} if missing)`, "~/pi-embodied")
		)?.trim();
		services = `${(checkout || "~/pi-embodied").replace(/^~(?=\/|$)/, homedir())}/services`;
	}

	let model: string | undefined;
	if (mode === "tools" || mode === "units") {
		const current = ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : undefined;
		const ids = ctx.modelRegistry.getAvailable().map((m) => `${m.provider}/${m.id}`);
		const opts = [...new Set([...(current ? [current] : []), ...ids])];
		model = opts.length
			? await ui.select("Planner model", opts)
			: (await ui.input("Planner model", "provider/model"))?.trim();
		if (!model?.includes("/")) return;
	}
	const weights =
		mode === "tools" && robot.weights ? await ui.confirm("Model weights", `Also download ${robot.weights}?`) : false;
	const answer = (await ui.input("Experiment directory", join(ctx.cwd, `embodied-${robot.id}`)))?.trim();
	const dir = resolve(ctx.cwd, (answer || join(ctx.cwd, `embodied-${robot.id}`)).replace(/^~(?=\/|$)/, homedir()));
	const choice: Choice = { robot, mode, remote, services, weights, model, dir };

	const file = join(dir, ".pi", "settings.json");
	let existing: Settings = {};
	if (!remote && existsSync(file)) {
		try {
			existing = JSON.parse(readFileSync(file, "utf8")) as Settings;
		} catch (e) {
			ui.notify(`${file} is not valid JSON: ${e instanceof Error ? e.message : String(e)}`, "error");
			return;
		}
	}
	const root = remote ? `${remote.checkout}/packages/embodied` : PKG;
	const source = remote ? root : packageSource(pi);
	const settings = experimentSettings(choice, source, root, existing);
	const text = `${JSON.stringify(settings, null, "\t")}\n`;
	const summary = [
		`${robot.label}, ${mode}${model ? `, planner ${model}` : ""}`,
		remote ? `services on ${remote.host}:${services}` : `services at ${services}`,
		`install: ${services}/setup.sh ${robot.id}${weights ? " --weights" : ""}${mode === "finetuned" ? ", the adapter download and the vLLM server" : ""}; the agent runs it after this confirmation (pi does not ask per step)`,
		remote ? "settings: the agent writes them on the box" : `writes ${file}`,
		`launch: cd ${dir} && pi ${launchFlags(choice)}`,
	].join("\n");
	if (!(await ui.confirm("Set up pi-embodied?", summary))) {
		ui.notify("Nothing written or installed.", "info");
		return;
	}
	if (!remote) {
		mkdirSync(dirname(file), { recursive: true });
		writeFileSync(file, text);
		ui.notify(`Wrote ${file}`, "info");
	}
	pi.sendUserMessage(handoff(choice, text), {
		expandPromptTemplates: true,
		...(ctx.isIdle() ? {} : { deliverAs: "followUp" as const }),
	});
}

export default function embodiedSetup(pi: ExtensionAPI) {
	pi.registerCommand(COMMAND, {
		description: "Set up pi-embodied: pick a robot, a mode, where services run and the planner; the agent installs",
		handler: (_args, ctx) => setup(pi, ctx),
	});
	pi.on("session_start", (_event, ctx) => {
		if (ctx.mode !== "tui" || !ctx.hasUI || robotActive(pi) || noticeSeen(ctx)) return;
		ctx.ui.notify(NOTICE, "info");
		pi.appendEntry(NOTICE_ENTRY, { shown: new Date().toISOString() });
	});
}
