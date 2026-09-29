/**
 * JSON a model wrote by hand, parsed leniently: what `JSON.parse` refuses only for its dialect, not
 * its content. One pass rewrites the text as strict JSON:
 * - a Markdown code fence around it, and `//` / `/* *\/` comments, are dropped;
 * - single-quoted strings become double-quoted ones (`\'` inside them is a quote);
 * - bare identifier keys (`{terminated: true}`) are quoted; Python's True / False / None are JSON's;
 * - trailing commas before `}` / `]` are dropped.
 * A document that is itself a JSON string holding JSON (`"{\"terminated\": true}"`), or whose quotes
 * were all backslash-escaped (`{\"terminated\": true}`), is unwrapped once. Anything still invalid
 * throws the strict parser's error with what was tried.
 */

/** `text` as JSON, repaired as above; `repaired` says whether the strict parser refused it. */
export function parseLenientJson(text: string): { value: unknown; repaired: boolean } {
	try {
		const value = JSON.parse(text);
		// A JSON string holding a JSON document: parse the document it holds.
		if (typeof value === "string" && /^\s*[[{]/.test(value))
			return { value: parseLenientJson(value).value, repaired: true };
		return { value, repaired: false };
	} catch (strict) {
		for (const candidate of candidates(text)) {
			try {
				let value = JSON.parse(toStrictJson(candidate));
				// A JSON string holding a JSON document: parse the document it holds.
				if (typeof value === "string" && /^\s*[[{]/.test(value)) value = JSON.parse(toStrictJson(value));
				return { value, repaired: true };
			} catch {}
		}
		throw new Error(`not JSON even leniently: ${strict instanceof Error ? strict.message : String(strict)}`);
	}
}

/** The text as written, then with a code fence stripped, then with escaped quotes unescaped. */
function candidates(text: string): string[] {
	const body = text.replace(/^﻿/, "").trim();
	const fenced = /^```[\w-]*\s*\n([\s\S]*?)\n?```\s*$/.exec(body);
	const unfenced = fenced ? fenced[1] : body;
	const out = [unfenced];
	// `{\"a\": 1}`: every quote escaped, as if the document had been pasted out of a JSON string.
	if (/\\"/.test(unfenced) && !/(^|[^\\])"/.test(unfenced.replace(/\\"/g, "")))
		out.push(
			unfenced
				.replace(/\\\\/g, "\u0000")
				.replace(/\\"/g, '"')
				.replace(/\u0000/g, "\\"),
		);
	return out;
}

const PYTHON: Record<string, string> = { True: "true", False: "false", None: "null" };
const ESCAPES: Record<string, string> = { n: "\n", t: "\t", r: "\r", b: "\b", f: "\f" };

/** One pass over `text` emitting strict JSON (see the module comment). */
export function toStrictJson(text: string): string {
	let out = "";
	let i = 0;
	const n = text.length;
	while (i < n) {
		const c = text[i];
		if (c === '"' || c === "'") {
			// A string, either quote: decoded, then written back as a JSON string (raw newlines escaped).
			let j = i + 1;
			let value = "";
			while (j < n && text[j] !== c) {
				if (text[j] === "\\" && j + 1 < n) {
					const e = text[j + 1];
					if (e === "u" && /^[0-9a-fA-F]{4}$/.test(text.slice(j + 2, j + 6))) {
						value += String.fromCharCode(Number.parseInt(text.slice(j + 2, j + 6), 16));
						j += 6;
						continue;
					}
					value += ESCAPES[e] ?? e;
					j += 2;
				} else value += text[j++];
			}
			out += JSON.stringify(value);
			i = j + 1;
		} else if (c === "/" && text[i + 1] === "/") {
			while (i < n && text[i] !== "\n") i++;
		} else if (c === "/" && text[i + 1] === "*") {
			const end = text.indexOf("*/", i + 2);
			i = end < 0 ? n : end + 2;
		} else if (c === ",") {
			// A trailing comma: the next significant character closes the container.
			let j = i + 1;
			for (;;) {
				while (j < n && /\s/.test(text[j])) j++;
				if (text[j] === "/" && text[j + 1] === "/") while (j < n && text[j] !== "\n") j++;
				else if (text[j] === "/" && text[j + 1] === "*") {
					const end = text.indexOf("*/", j + 2);
					j = end < 0 ? n : end + 2;
				} else break;
			}
			if (text[j] !== "}" && text[j] !== "]") out += c;
			i++;
		} else if (/[A-Za-z_$]/.test(c)) {
			let j = i;
			while (j < n && /[\w$]/.test(text[j])) j++;
			const word = text.slice(i, j);
			let k = j;
			while (k < n && /\s/.test(text[k])) k++;
			if (text[k] === ":") out += JSON.stringify(word);
			else out += PYTHON[word] ?? word;
			i = j;
		} else {
			out += c;
			i++;
		}
	}
	return out;
}
