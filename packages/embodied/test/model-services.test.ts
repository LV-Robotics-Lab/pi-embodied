import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createServer as createHttpServer } from "node:http";
import { type AddressInfo, createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { gpuFreeMiB, lockFile, type ModelService, modelServices, servicePort } from "../src/model-services.ts";
import { defineRobot } from "../src/robot.ts";
import { RpcClient } from "../src/rpc.ts";

const PYTHON = process.env.PYTHON ?? "python3";

/** A services dir with a stand-in model server: healthz, shutdown, --parent-watch; `--fail` exits at load. */
function servicesDir(): string {
	const dir = mkdtempSync(join(tmpdir(), "model-services-"));
	writeFileSync(
		join(dir, "fake_model.py"),
		`import argparse, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
p = argparse.ArgumentParser()
for f in ("--host", "--transport", "--cuda-device", "--embodiment"):
    p.add_argument(f)
p.add_argument("--port", type=int)
p.add_argument("--parent-watch", action="store_true")
p.add_argument("--fail", action="store_true")
p.add_argument("--load", type=float, default=0.0)
a = p.parse_args()
print("loading", flush=True)
if a.fail:
    print("RuntimeError: MODEL_CHECKPOINT_PATH is not set", flush=True)
    sys.exit(3)
time.sleep(a.load)
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps({"ok": True, "result": {"argv": sys.argv[1:], "env": os.environ.get("FAKE_ENV")}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        if req["method"] == "shutdown":
            threading.Timer(0.1, lambda: os._exit(0)).start()
    def log_message(self, *_):
        pass
srv = HTTPServer(("127.0.0.1", a.port), H)
if a.parent_watch:
    threading.Thread(target=lambda: (sys.stdin.read(), os._exit(0)), daemon=True).start()
srv.serve_forever()
`,
	);
	return dir;
}

async function freePort(): Promise<number> {
	const s = createServer();
	await new Promise<void>((r) => s.listen(0, "127.0.0.1", r));
	const { port } = s.address() as AddressInfo;
	await new Promise((r) => s.close(r));
	return port;
}

function stubPi(flags: Record<string, string>) {
	const values: Record<string, unknown> = {};
	const handlers = new Map<string, ((e: unknown, ctx: unknown) => unknown)[]>();
	const pi = {
		registerFlag: (name: string, o: { default?: unknown }) => {
			values[name] = name in flags ? flags[name] : o.default;
		},
		getFlag: (name: string) => (name in values ? values[name] : flags[name]),
		on: (name: string, fn: (e: unknown, ctx: unknown) => unknown) =>
			handlers.set(name, [...(handlers.get(name) ?? []), fn]),
		registerTool: () => {},
		registerCommand: () => {},
		registerProvider: () => {},
		setActiveTools: () => {},
		getActiveTools: () => [],
		appendEntry: () => {},
		events: { emit: () => {}, on: () => () => {} },
	} as unknown as ExtensionAPI;
	const emit = async (name: string, ctx: unknown) => {
		for (const fn of handlers.get(name) ?? []) await fn({ type: name }, ctx);
	};
	return { pi, values, emit };
}

const fake = (name: string, extra: string[] = []): ModelService => ({
	name,
	flag: name,
	module: "fake_model",
	args: () => extra,
	env: () => ({ FAKE_ENV: name }),
});

const healthy = (port: number) =>
	new RpcClient(`http://127.0.0.1:${port}`).call("healthz", {}, 1_000).then(
		() => true,
		() => false,
	);

test("servicePort takes only loopback http URLs with a port", () => {
	assert.equal(servicePort("sam3", "http://127.0.0.1:18300"), 18300);
	assert.equal(servicePort("sam3", "localhost:18301"), 18301);
	assert.throws(() => servicePort("molmo", "off"), /--molmo off/);
	assert.throws(() => servicePort("vla", "http://gpu-box:18200"), /127\.0\.0\.1/);
	assert.throws(() => servicePort("lingbot", "ws://127.0.0.1:18400"), /127\.0\.0\.1/);
});

test("off: nothing starts", async () => {
	const { pi } = stubPi({});
	const ms = modelServices(pi, { models: [fake("sam3")] });
	await ms.start();
	assert.deepEqual(ms.running(), []);
});

test("starts every listed service on its flag's port, waits for healthz, and stops them", async () => {
	const [a, b] = [await freePort(), await freePort()];
	const logs = mkdtempSync(join(tmpdir(), "model-services-logs-"));
	const { pi } = stubPi({
		services: servicesDir(),
		python: PYTHON,
		sam3: `http://127.0.0.1:${a}`,
		vla: `http://127.0.0.1:${b}`,
		"serve-models": "all",
		"serve-cuda-device": "1",
		"serve-log-dir": logs,
	});
	const ms = modelServices(pi, { models: [fake("sam3"), fake("vla", ["--embodiment", "libero", "--load", "1.5"])] });
	await ms.start();
	try {
		assert.deepEqual(ms.running().sort(), ["sam3", "vla"]);
		const info = await new RpcClient(`http://127.0.0.1:${b}`).call<{ argv: string[]; env: string }>("healthz");
		assert.deepEqual(info.argv.slice(0, 4), ["--embodiment", "libero", "--load", "1.5"]);
		assert.ok(
			info.argv.join(" ").includes(`--cuda-device 1 --transport http --host 127.0.0.1 --port ${b} --parent-watch`),
		);
		assert.equal(info.env, "vla");
		assert.match(readFileSync(join(logs, `sam3-${a}.log`), "utf8"), /loading/);
	} finally {
		await ms.stop();
	}
	assert.deepEqual(ms.running(), []);
	assert.equal(await healthy(a), false);
	assert.equal(await healthy(b), false);
});

test("one service failing stops the others and fails with its log", async () => {
	const [a, b] = [await freePort(), await freePort()];
	const { pi } = stubPi({
		services: servicesDir(),
		python: PYTHON,
		sam3: `http://127.0.0.1:${a}`,
		molmo: `http://127.0.0.1:${b}`,
		"serve-models": "sam3,molmo",
		"serve-log-dir": mkdtempSync(join(tmpdir(), "model-services-logs-")),
	});
	const ms = modelServices(pi, { models: [fake("sam3", ["--load", "3"]), fake("molmo", ["--fail"])] });
	await assert.rejects(
		ms.start(),
		/--serve-models molmo on :\d+: exited \(3\).*\n[\s\S]*MODEL_CHECKPOINT_PATH is not set/,
	);
	assert.deepEqual(ms.running(), []);
	assert.equal(await healthy(a), false);
});

test("refuses unknown names, a --serve-python for a service it does not start, and a port that already serves", async () => {
	const other = createHttpServer((_req, res) => res.end(JSON.stringify({ ok: true, result: {} })));
	await new Promise<void>((r) => other.listen(0, "127.0.0.1", r));
	const port = (other.address() as AddressInfo).port;
	try {
		const models = [fake("sam3"), fake("molmo")];
		const flags = { services: servicesDir(), python: PYTHON, sam3: `http://127.0.0.1:${port}`, molmo: "off" };
		const unknown = modelServices(stubPi({ ...flags, "serve-models": "sam3,vla" }).pi, { models });
		await assert.rejects(unknown.start(), /unknown vla; this robot serves sam3, molmo/);
		const py = modelServices(stubPi({ ...flags, "serve-models": "sam3", "serve-python": "molmo=/x/python" }).pi, {
			models,
		});
		await assert.rejects(py.start(), /--serve-python: molmo not in --serve-models/);
		const off = modelServices(stubPi({ ...flags, "serve-models": "molmo" }).pi, { models });
		await assert.rejects(off.start(), /--molmo off/);
		const busy = modelServices(stubPi({ ...flags, "serve-models": "sam3" }).pi, { models });
		await assert.rejects(busy.start(), /already serves; drop it from --serve-models to attach to it/);
		assert.deepEqual(busy.running(), []);
	} finally {
		other.close();
	}
});

test("the robot base starts the services before the robot attaches, and fails closed when they fail", async () => {
	const port = await freePort();
	const { pi, emit } = stubPi({
		services: servicesDir(),
		python: PYTHON,
		sam3: `http://127.0.0.1:${port}`,
		"serve-models": "sam3",
		"serve-log-dir": mkdtempSync(join(tmpdir(), "model-services-logs-")),
	});
	let seen: boolean | undefined;
	defineRobot(pi, {
		name: "toy",
		task: [],
		keepImages: 1,
		services: { models: [fake("sam3")] },
		start: async () => {
			seen = await healthy(port);
			return [];
		},
		result: () => ({}),
		finish: {
			description: "finish",
			parameters: Type.Object({ status: Type.String(), summary: Type.String() }),
			result: () => ({ content: [], details: {} }),
		},
	});
	const log = console.error;
	const errors: string[] = [];
	console.error = (line: string) => errors.push(line);
	let shutdown = false;
	const ctx = {
		hasUI: false,
		cwd: tmpdir(),
		ui: { notify: () => {} },
		shutdown: () => {
			shutdown = true;
		},
		sessionManager: { getBranch: () => [], getSessionDir: () => tmpdir() },
	};
	try {
		await emit("session_start", ctx);
		assert.equal(seen, true, errors.join("\n"));
		assert.equal(shutdown, false);
		await emit("session_shutdown", ctx);
		assert.equal(await healthy(port), false);
	} finally {
		console.error = log;
		process.exitCode = undefined;
	}
});

test("--serve-min-free waits for the GPU's free memory and gives up at --serve-timeout", async () => {
	const bin = mkdtempSync(join(tmpdir(), "model-services-smi-"));
	const smi = join(bin, "nvidia-smi");
	const report = (used: number) => writeFileSync(smi, `#!/bin/sh\necho "${used}, 32607"\n`, { mode: 0o755 });
	const path = process.env.PATH;
	process.env.PATH = `${bin}:${path}`;
	try {
		report(30000);
		assert.equal(await gpuFreeMiB("1"), 2607);
		const flags = {
			services: servicesDir(),
			python: PYTHON,
			sam3: `http://127.0.0.1:${await freePort()}`,
			"serve-models": "sam3",
			"serve-cuda-device": "1",
			"serve-min-free": "8000",
			"serve-timeout": "2",
			"serve-log-dir": mkdtempSync(join(tmpdir(), "model-services-logs-")),
		};
		const short = modelServices(stubPi(flags).pi, { models: [fake("sam3")] });
		await assert.rejects(short.start(), /GPU 1 has 2607 MiB free, not 8000/);
		assert.deepEqual(short.running(), []);
		report(1000);
		const roomy = modelServices(stubPi(flags).pi, { models: [fake("sam3")] });
		await roomy.start();
		assert.deepEqual(roomy.running(), ["sam3"]);
		await roomy.stop();
	} finally {
		process.env.PATH = path;
	}
});

const hasFlock = (() => {
	try {
		execFileSync("flock", ["--version"], { stdio: "ignore" });
		return true;
	} catch {
		return false;
	}
})();

test("--serve-lock holds an exclusive flock until released", { skip: !hasFlock && "no flock(1) here" }, async () => {
	const path = join(mkdtempSync(join(tmpdir(), "model-services-lock-")), "gpu1.lock");
	const first = await lockFile(path, 5);
	await assert.rejects(lockFile(path, 1), /not acquired within 1 s/);
	first.release();
	const second = await lockFile(path, 5);
	second.release();
});
