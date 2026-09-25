import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { createServer as createHttpServer } from "node:http";
import { type AddressInfo, createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
	formatTable,
	parseFlags,
	pathRows,
	plannerRow,
	portRow,
	probeEndpoint,
	type RobotCheckSpec,
	runChecks,
} from "../src/check.ts";

/** A services-style RPC server answering healthz with `ok`. */
async function rpcServer(ok: boolean) {
	const server = createHttpServer((req, res) => {
		let body = "";
		req.on("data", (c) => {
			body += c;
		});
		req.on("end", () => {
			if (req.url === "/models") return res.writeHead(401).end("{}");
			const { method } = JSON.parse(body) as { method: string };
			res.writeHead(ok ? 200 : 500, { "Content-Type": "application/json" });
			res.end(JSON.stringify(ok ? { ok: true, result: { method } } : { ok: false, error: "model not loaded" }));
		});
	});
	await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
	const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
	return {
		url,
		close: () => {
			server.closeAllConnections();
			server.close();
		},
	};
}

async function closedPort() {
	const srv = createServer();
	await new Promise<void>((r) => srv.listen(0, "127.0.0.1", r));
	const { port } = srv.address() as AddressInfo;
	await new Promise((r) => srv.close(r));
	return port;
}

const python = (() => {
	for (const p of ["python3", "python"])
		try {
			execFileSync(p, ["-c", "import sys; assert sys.version_info >= (3, 10)"], { stdio: "ignore" });
			return p;
		} catch {}
	return undefined;
})();

test("parseFlags reads --k v, --k=v and bare booleans, skipping positionals and short flags", () => {
	const f = parseFlags([
		"libero",
		"-e",
		"x",
		"--units",
		"--python=/v/bin/python",
		"--vla",
		"http://h:1",
		"--dashboard",
	]);
	assert.deepEqual(f, { units: true, python: "/v/bin/python", vla: "http://h:1", dashboard: true });
});

test("pathRows: required paths fail, optional ones warn, flags override the variable", () => {
	const dir = mkdtempSync(join(tmpdir(), "check-"));
	mkdirSync(join(dir, "robolab"));
	writeFileSync(join(dir, "ckpt.pt"), "x");
	const rows = pathRows(
		[
			{ env: "A_DIR", kind: "dir", required: true, why: "a", contains: "robolab" },
			{ env: "B_FILE", kind: "file", required: false, why: "b" },
			{ env: "C_DIR", kind: "dir", required: true, why: "c" },
			{ env: "D_DIR", flag: "assets", kind: "dir", required: true, why: "d" },
			{ env: "E_FILE", kind: "file", required: true, why: "e" },
		],
		{ assets: dir },
		{ A_DIR: dir, B_FILE: join(dir, "missing.pt"), E_FILE: dir },
	);
	assert.deepEqual(
		rows.map((r) => [r.check, r.status]),
		[
			["$A_DIR", "PASS"],
			["$B_FILE", "WARN"],
			["$C_DIR", "FAIL"],
			["--assets", "PASS"],
			["$E_FILE", "FAIL"],
		],
	);
	assert.match(rows[2].detail, /not set/);
	assert.match(rows[4].detail, /is not a file/);
});

test("probeEndpoint: healthz ok, a server error, a closed port, and ws:// by TCP connect", async () => {
	const up = await rpcServer(true);
	const sick = await rpcServer(false);
	try {
		assert.equal((await probeEndpoint(up.url)).ok, true);
		const bad = await probeEndpoint(sick.url);
		assert.equal(bad.ok, false);
		assert.match(bad.detail, /HTTP 500 model not loaded/);
		const port = await closedPort();
		const down = await probeEndpoint(`http://127.0.0.1:${port}`);
		assert.deepEqual(down, { ok: false, detail: "ECONNREFUSED" });
		assert.equal((await probeEndpoint(up.url.replace("http:", "ws:"))).ok, true);
		assert.equal((await probeEndpoint(`ws://127.0.0.1:${port}`)).ok, false);
	} finally {
		up.close();
		sick.close();
	}
});

test("portRow reports a port in use", async () => {
	const srv = createServer();
	await new Promise<void>((r) => srv.listen(0, "127.0.0.1", r));
	const { port } = srv.address() as AddressInfo;
	try {
		assert.equal((await portRow(port)).status, "FAIL");
	} finally {
		srv.close();
	}
	assert.equal((await portRow(port)).status, "PASS");
	assert.equal((await portRow(0)).status, "SKIP");
});

test("plannerRow looks the provider up in models.json; 401 still means the gateway is up", async () => {
	const gw = await rpcServer(true);
	const dir = mkdtempSync(join(tmpdir(), "check-"));
	const models = join(dir, "models.json");
	const port = await closedPort();
	writeFileSync(
		models,
		JSON.stringify({ providers: { relay: { baseUrl: gw.url }, dead: { baseUrl: `http://127.0.0.1:${port}/v1` } } }),
	);
	try {
		const ok = await plannerRow("relay/m", undefined, models);
		assert.equal(ok.status, "PASS");
		assert.match(ok.detail, /HTTP 401 \(up; needs the key\)/);
		assert.equal((await plannerRow("dead/m", undefined, models)).status, "FAIL");
		assert.equal((await plannerRow("nobody/m", undefined, models)).status, "SKIP");
		assert.equal((await plannerRow(undefined)).status, "SKIP");
	} finally {
		gw.close();
	}
});

test(
	"runChecks: imports through the services Python, tools-only servers soften with --units",
	{ skip: !python },
	async () => {
		const spec: RobotCheckSpec = {
			python: { flag: "python", env: [] },
			imports: ["json", "pi_embodied_services", "no_such_module_xyz"],
			find: ["os", "no_such_package_xyz"],
			endpoints: [
				{ flag: "vla", default: `http://127.0.0.1:${await closedPort()}`, why: "VLA", toolsOnly: true },
				{ flag: "rldx", why: "RLDX" },
			],
			gpu: false,
		};
		const rows = await runChecks("fake", { python: python as string, units: "true" }, { spec, timeoutMs: 60_000 });
		const status = Object.fromEntries(rows.map((r) => [r.check, r.status]));
		assert.equal(status.services, "PASS");
		assert.equal(status.python, "PASS");
		assert.equal(status["import json"], "PASS");
		assert.equal(status["import pi_embodied_services"], "PASS");
		assert.equal(status["import no_such_module_xyz"], "FAIL");
		assert.equal(status["find os"], "PASS");
		assert.equal(status["find no_such_package_xyz"], "FAIL");
		assert.equal(status["--vla"], "WARN");
		assert.equal(status["--rldx"], "SKIP");
		assert.equal(status.planner, "SKIP");
		const table = formatTable(rows);
		assert.match(table, /^NOT READY: \d+ pass, 1 warn, 2 fail, \d+ skip$/m);

		const missing = await runChecks("fake", { python: "/nonexistent/python" }, { spec, timeoutMs: 10_000 });
		assert.match(missing.find((r) => r.check === "python")?.detail ?? "", /not found/);
		assert.equal(missing.find((r) => r.check === "--vla")?.status, "FAIL");
	},
);

test("runChecks refuses an unknown robot", async () => {
	const rows = await runChecks("nosuchrobot", {});
	assert.equal(rows[0].status, "FAIL");
	assert.match(rows[0].detail, /unknown robot/);
});
