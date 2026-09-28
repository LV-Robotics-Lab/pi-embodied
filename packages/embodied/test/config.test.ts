import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	configProblem,
	cudaDevice,
	deploymentRecord,
	dir,
	python,
	SERVICE_DEFAULTS,
	service,
	servicesDir,
} from "../src/infra/config.ts";
import { useDeployment } from "./helpers/deployment.ts";

const pi = (flags: Record<string, unknown> = {}) => ({ getFlag: (n: string) => flags[n] }) as unknown as ExtensionAPI;

/** Run `fn` with a global config file holding `global` and a project dir (cwd) holding `project`. */
function withFiles(global: unknown, project: unknown, fn: () => void) {
	const root = mkdtempSync(join(tmpdir(), "pi-embodied-cfg-"));
	const g = join(root, "global.json");
	writeFileSync(g, JSON.stringify(global));
	mkdirSync(join(root, "proj", ".pi"), { recursive: true });
	if (project !== undefined) writeFileSync(join(root, "proj", ".pi", "embodied.json"), JSON.stringify(project));
	const cwd = process.cwd();
	const env = process.env.PI_EMBODIED_CONFIG;
	process.env.PI_EMBODIED_CONFIG = g;
	process.chdir(join(root, "proj"));
	try {
		fn();
	} finally {
		process.chdir(cwd);
		process.env.PI_EMBODIED_CONFIG = env;
	}
}

test("built-in < deployment; the project file wins key by key; --deployment picks one", () => {
	withFiles(
		{
			deployments: {
				box: { services: { sam3: "http://g:1", molmo: "http://g:2" }, python: { default: "/g/python" } },
				other: { services: { sam3: "http://o:1" } },
			},
		},
		{ deployments: { box: { services: { sam3: "http://p:1" }, python: { libero: "/p/libero" } } } },
		() => {
			const p = pi({ deployment: "box" });
			assert.equal(configProblem(p), undefined);
			assert.equal(service(p, "sam3"), "http://p:1");
			assert.equal(service(p, "molmo"), "http://g:2");
			assert.equal(service(p, "vla"), SERVICE_DEFAULTS.vla);
			assert.equal(service(p, "ik"), "");
			assert.equal(python(p, "libero"), "/p/libero");
			assert.equal(python(p, "robosuite"), "/g/python");
			assert.equal(service(pi({ deployment: "other" }), "sam3"), "http://o:1");
			// Two deployments and no `default`: none is picked, the built-ins apply.
			assert.equal(service(pi(), "sam3"), SERVICE_DEFAULTS.sam3);
			assert.deepEqual(deploymentRecord(pi()), { deployment: "", deployment_sha: null });
			const rec = deploymentRecord(p);
			assert.equal(rec.deployment, "box");
			assert.match(rec.deployment_sha ?? "", /^[0-9a-f]{16}$/);
		},
	);
});

test("an unknown deployment, service key or dir kind, or a malformed file, is a start-stopping problem", () => {
	withFiles({ deployments: { a: {}, b: {} } }, undefined, () => {
		assert.match(configProblem(pi({ deployment: "c" })) ?? "", /--deployment c is not in .*known: a, b/);
	});
	withFiles({ deployments: { default: { services: { sam: "x" } } } }, undefined, () => {
		assert.match(configProblem(pi()) ?? "", /unknown services\.sam/);
	});
	withFiles({ deployments: { default: { dirs: { log: "x" } } } }, undefined, () => {
		assert.match(configProblem(pi()) ?? "", /unknown dirs\.log/);
	});
	const root = mkdtempSync(join(tmpdir(), "pi-embodied-cfg-"));
	writeFileSync(join(root, "bad.json"), "{ nope");
	const env = process.env.PI_EMBODIED_CONFIG;
	process.env.PI_EMBODIED_CONFIG = join(root, "bad.json");
	try {
		assert.match(configProblem(pi()) ?? "", /bad\.json/);
	} finally {
		process.env.PI_EMBODIED_CONFIG = env;
	}
});

test("the eval scripts' per-worker environment wins over the deployment; the only deployment is the default", () => {
	useDeployment({ cuda_device: "0", dirs: { logs: "/cfg/logs" }, services_dir: "/cfg/services" });
	const saved = { ...process.env };
	try {
		assert.equal(cudaDevice(pi()), "0");
		assert.equal(dir(pi(), "logs"), "/cfg/logs");
		assert.equal(dir(pi(), "video", "/fallback"), "/fallback");
		assert.equal(servicesDir(pi()), "/cfg/services");
		process.env.PI_EMBODIED_CUDA_DEVICE = "1";
		process.env.PI_EMBODIED_DIRS_LOGS = "/episode";
		assert.equal(cudaDevice(pi()), "1");
		assert.equal(dir(pi(), "logs"), "/episode");
	} finally {
		process.env.PI_EMBODIED_CUDA_DEVICE = saved.PI_EMBODIED_CUDA_DEVICE;
		process.env.PI_EMBODIED_DIRS_LOGS = saved.PI_EMBODIED_DIRS_LOGS;
		if (saved.PI_EMBODIED_CUDA_DEVICE === undefined) delete process.env.PI_EMBODIED_CUDA_DEVICE;
		if (saved.PI_EMBODIED_DIRS_LOGS === undefined) delete process.env.PI_EMBODIED_DIRS_LOGS;
	}
});
