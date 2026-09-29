import assert from "node:assert/strict";
import { test } from "node:test";
import { parseLenientJson } from "../src/infra/lenient-json.ts";

test("hand-written JSON is repaired only in its dialect: comments, quotes, fences, trailing commas, Python literals", () => {
	const ok = (text: string, value: unknown) =>
		assert.deepEqual(parseLenientJson(text), { value, repaired: true }, text);
	ok('// audit of goal_t3_s7\n{"terminated": true, "task_language": "put the bowl"}', {
		terminated: true,
		task_language: "put the bowl",
	});
	ok("```json\n{'terminated': True, 'task_language': 'it\\'s the \"bowl\"', seed: 0, notes: None,}\n```", {
		terminated: true,
		task_language: 'it\'s the "bowl"',
		seed: 0,
		notes: null,
	});
	// Every quote escaped, as if pasted out of a JSON string; one layer of backslashes comes off.
	ok('{\\"terminated\\": true, \\"path\\": \\"a\\\\\\\\b\\"}', { terminated: true, path: "a\\b" });
	// A JSON string holding the document.
	ok(JSON.stringify(JSON.stringify({ terminated: true })), { terminated: true });
	// Comment markers and brackets inside strings are text; a raw newline in a string is escaped.
	ok('{"a": [1, 2, /* two */ ], "b": "x // y", "c": \'z, ]\', "d": "l1\nl2"}', {
		a: [1, 2],
		b: "x // y",
		c: "z, ]",
		d: "l1\nl2",
	});
	assert.deepEqual(parseLenientJson('{"a": 1}'), { value: { a: 1 }, repaired: false });
	assert.throws(() => parseLenientJson("{terminated: }"), /not JSON even leniently: /);
	assert.throws(() => parseLenientJson("the episode was solved"), /not JSON even leniently/);
});
