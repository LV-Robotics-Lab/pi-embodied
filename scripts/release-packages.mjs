import { readFileSync } from "node:fs";
import { join } from "node:path";
import { findPackageDirectories } from "./package-workspaces.mjs";

/** Scopes of public packages in this repository that are versioned and published on their own (e.g. @lv-robotics/pi-embodied). */
const separatelyReleasedScopes = ["@lv-robotics/"];

/**
 * Whether a package is part of pi's lockstep release: public and not in a separately released scope.
 * Other public packages in this repository (e.g. @lv-robotics/pi-embodied) are versioned and
 * published on their own.
 */
export function isPiReleasePackage(pkg) {
	return (
		pkg.private !== true &&
		typeof pkg.name === "string" &&
		!separatelyReleasedScopes.some((scope) => pkg.name.startsWith(scope))
	);
}

export function getPublicWorkspacePackages(root) {
	return findPackageDirectories(root)
		.map((directory) => ({
			directory,
			...JSON.parse(readFileSync(join(directory, "package.json"), "utf8")),
		}))
		.filter((pkg) => pkg.private !== true)
		.map(({ directory, name, version }) => ({ directory, name, version }));
}

/** The public packages released in lockstep with pi (see `isPiReleasePackage`). */
export function getPiReleasePackages(root) {
	return getPublicWorkspacePackages(root).filter(isPiReleasePackage);
}
