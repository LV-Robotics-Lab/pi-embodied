import { readFileSync } from "node:fs";
import { join } from "node:path";
import { findPackageDirectories } from "./package-workspaces.mjs";

/**
 * Whether a package is part of pi's lockstep release: public and in the @earendil-works scope.
 * Other public packages in this repository (e.g. @lv-robotics/pi-embodied) are versioned and
 * published on their own.
 */
export function isPiReleasePackage(pkg) {
	return pkg.private !== true && typeof pkg.name === "string" && pkg.name.startsWith("@earendil-works/");
}

export function getPublicWorkspacePackages() {
	return findPackageDirectories()
		.map((directory) => ({
			directory,
			...JSON.parse(readFileSync(join(directory, "package.json"), "utf8")),
		}))
		.filter((pkg) => pkg.private !== true)
		.map(({ directory, name, version }) => ({ directory, name, version }));
}

/** The public packages released in lockstep with pi (see `isPiReleasePackage`). */
export function getPiReleasePackages() {
	return getPublicWorkspacePackages().filter(isPiReleasePackage);
}
