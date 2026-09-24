import type { Robot } from "../toolkit.ts";

/** Robots shipped with the harness, by name. */
const ROBOTS: Record<string, (endpoint: string) => Robot> = {};

export function makeRobot(name: string, endpoint: string): Robot {
	const factory = ROBOTS[name];
	if (!factory) {
		const available = Object.keys(ROBOTS);
		throw new Error(`unknown robot ${name}; available: ${available.length ? available.join(", ") : "(none yet)"}`);
	}
	return factory(endpoint);
}
