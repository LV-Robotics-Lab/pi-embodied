import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { ROBOT_VIEWS } from "../src/finetuned/index.ts";
import { convertRun } from "../src/finetuned/prepare.ts";
import { resolveViews, transformAll } from "../src/finetuned/transform.ts";
import { decodePng, fingerprint, formatView, parseView } from "../src/finetuned/views.ts";
import { encodePng } from "../src/png.ts";

/** The synthetic frames of finetuned.test.ts's reference: (x*3+y+seed, x*y+seed, x^y) mod 256. */
function pattern(h: number, w: number, seed: number) {
	const rgb = Buffer.alloc(h * w * 3);
	for (let y = 0; y < h; y++)
		for (let x = 0; x < w; x++) {
			const o = (y * w + x) * 3;
			rgb[o] = (x * 3 + y + seed) % 256;
			rgb[o + 1] = (x * y + seed) % 256;
			rgb[o + 2] = (x ^ y) & 255;
		}
	return { width: w, height: h, rgb };
}
const agentRaw = pattern(512, 512, 7);
const wristRaw = pattern(480, 640, 11);
const VIEWS = { agentview: "square=256", wrist: "rot=90,flip=vertical,crop=1.3333,square=256" };
const png = (img: { width: number; height: number; rgb: Buffer }) => encodePng(img.rgb, img.width, img.height);
const TRANSFORM = new URL("../src/finetuned/transform.ts", import.meta.url).pathname;

test("transform.ts writes the provider's pixels: the reference fingerprints, byte-identical to prepare.ts", () => {
	const dir = mkdtempSync(join(tmpdir(), "transform-"));
	writeFileSync(join(dir, "a.png"), png(agentRaw));
	writeFileSync(join(dir, "w.png"), png(wristRaw));
	const manifest = {
		views: VIEWS,
		jobs: [
			{ src: join(dir, "a.png"), dst: join(dir, "out", "agentview", "0000.png"), camera: "agentview" },
			{ src: join(dir, "w.png"), dst: join(dir, "out", "wrist", "0000.png"), camera: "wrist" },
		],
	};
	writeFileSync(join(dir, "m.json"), JSON.stringify(manifest));
	const reply = JSON.parse(
		execFileSync(process.execPath, ["--experimental-strip-types", TRANSFORM, join(dir, "m.json")], {
			encoding: "utf8",
			stdio: ["ignore", "pipe", "pipe"],
		}).trim(),
	);
	assert.deepEqual(reply, {
		count: 2,
		views: { agentview: "rot=0,flip=none,square=256", wrist: "rot=90,flip=vertical,crop=1.3333,square=256" },
	});
	const outA = readFileSync(manifest.jobs[0].dst);
	const outW = readFileSync(manifest.jobs[1].dst);
	// The fingerprints Show-Harness's own prepare_view produced for these frames (finetuned.test.ts REF).
	assert.equal(fingerprint(decodePng(outA)).sha1, "6c883980c15b");
	assert.equal(fingerprint(decodePng(outW)).sha1, "1e53f10798cd");
	// And the very bytes prepare.ts writes for a GUMI step of the same frames.
	const run = join(dir, "run");
	mkdirSync(join(run, "images", "agentview"), { recursive: true });
	mkdirSync(join(run, "images", "wrist"), { recursive: true });
	writeFileSync(join(run, "images", "agentview", "0000.png"), png(agentRaw));
	writeFileSync(join(run, "images", "wrist", "0000.png"), png(wristRaw));
	const row = { step: 0, token: "GRASP", agentview: "images/agentview/0000.png", wrist: "images/wrist/0000.png" };
	writeFileSync(join(run, "actions.jsonl"), `${JSON.stringify(row)}\n`);
	const views = { agentview: parseView(VIEWS.agentview), wrist: parseView(VIEWS.wrist) };
	const r = convertRun(run, join(dir, "prepared"), 0, views, "t", {});
	assert.ok(readFileSync(join(r.out, "agentview", "0000.png")).equals(outA));
	assert.ok(readFileSync(join(r.out, "wrist", "0000.png")).equals(outW));
});

test("a manifest names the robot's calibrated transform or explicit view specs", () => {
	const libero = resolveViews({ robot: "libero" });
	assert.equal(formatView(libero.wrist), formatView(ROBOT_VIEWS.libero.wrist));
	assert.equal(formatView(resolveViews({ robot: "maniskill", env_id: "BlockPAP-v1" }).wrist), "rot=0,flip=none");
	assert.equal(formatView(resolveViews({ views: VIEWS }).wrist), VIEWS.wrist);
	assert.throws(() => resolveViews({ robot: "nope" }), /no camera transform for robot "nope"/);
	assert.throws(() => resolveViews({ views: { agentview: "raw" } }), /needs agentview and wrist/);
	const dir = mkdtempSync(join(tmpdir(), "transform-"));
	writeFileSync(join(dir, "a.png"), png(pattern(8, 6, 1)));
	assert.equal(transformAll([{ src: join(dir, "a.png"), dst: join(dir, "b.png"), camera: "agentview" }], libero), 1);
	assert.deepEqual(fingerprint(decodePng(readFileSync(join(dir, "b.png")))).shape, [256, 256, 3]);
});
