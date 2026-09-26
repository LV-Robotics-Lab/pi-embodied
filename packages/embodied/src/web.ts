/**
 * Web access for robots (--web-tools): OpenETA's `web_search` / `web_fetch`
 * (agent/tools/web_access.py) come from pi packages, not from this package:
 *
 *   pi install npm:pi-web-search             # web_search: the provider's native search (OpenAI Responses
 *                                            # web_search, Gemini grounding, Anthropic, xAI), as OpenETA does
 *   pi install npm:@zeldrisho/pi-web-fetch   # web_fetch: bounded public http(s) pages as Markdown, SSRF-checked
 *
 * A robot activates only its own tools at start (../robot.ts), which would hide the packages' tools;
 * --web-tools keeps `web_search` and `web_fetch` active next to them. Both must be installed: a
 * missing one fails the start with the install command (nothing silently drops). Their results are
 * untrusted page content, as the packages say.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export const WEB_TOOLS: Record<string, string> = {
	web_search: "npm:pi-web-search",
	web_fetch: "npm:@zeldrisho/pi-web-fetch",
};

export function webTools(pi: ExtensionAPI) {
	pi.registerFlag("web-tools", {
		type: "boolean",
		default: false,
		description: `Keep web_search and web_fetch (pi install ${Object.values(WEB_TOOLS).join(", ")}) active for the robot`,
	});
	return {
		/** The web tools to activate; throws when --web-tools is on and a package's tool is not loaded. */
		tools(): string[] {
			if (pi.getFlag("web-tools") !== true) return [];
			const loaded = new Set(pi.getAllTools().map((t) => t.name));
			const missing = Object.keys(WEB_TOOLS).filter((t) => !loaded.has(t));
			if (missing.length)
				throw new Error(
					`--web-tools: ${missing.join(", ")} not loaded; pi install ${missing.map((t) => WEB_TOOLS[t]).join(" and ")}`,
				);
			return Object.keys(WEB_TOOLS);
		},
	};
}
