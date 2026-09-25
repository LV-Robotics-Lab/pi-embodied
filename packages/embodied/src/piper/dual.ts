/**
 * Both AgileX Piper arms of the Show-Harness dual rig (Cobot Magic), for pi:
 *
 *   pi -e packages/embodied/src/piper/dual.ts --operator --task banana_handover --robot-config my_dual.yaml
 *   pi -e packages/embodied/src/piper/dual.ts --operator --units=true ...   (one `act` per arm and step, STILL)
 *
 * The env server config needs the per-arm `arms:` block
 * (services/pi_embodied_services/robots/piper/config/dual_example.yaml); see ./index.ts.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { piperRobot } from "./index.ts";

export default function piperDual(pi: ExtensionAPI) {
	piperRobot(pi, true);
}
