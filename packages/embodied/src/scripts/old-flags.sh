# Sourced by the eval scripts: `old_flags "$@" || exit 2` refuses a flag that was renamed or moved to
# the deployment config before any cell runs (there are no aliases; docs/flags-migration.md). pi itself
# would stop each episode with "Unknown option", which an eval script would record cell by cell.
old_flags() {
	local a name new
	for a in "$@"; do
		case $a in --*) ;; *) continue ;; esac
		name=${a%%=*}
		case $name in
		--env | --robot-env) new="--env-url" ;;
		--task-name | --env-id) new="--task" ;;
		--robot | --arm-id) new="--arm" ;;
		--robot-backend) new="--backend" ;;
		--eval-seed) new="--layout-set" ;;
		--robot-cameras) new="--cameras" ;;
		--units-vlm-model | --vdm-model | --attach-vlm-model) new="--aux-model (one role: aux.<role> in the deployment config)" ;;
		--sam3 | --molmo | --rldx | --lingbot | --ft-endpoint | --robot-sam3 | --robot-vla) new="services.<name> in the deployment config (docs/flags-migration.md)" ;;
		--openvla | --openvla-oft | --gr00t) new="--vla-adapter ${name#--} and services.<name>" ;;
		--contact-graspnet | --graspgenx | --anygrasp | --graspnet1b) new="--grasp <backend> and services.<backend>" ;;
		--anyplace) new="--place anyplace and services.anyplace" ;;
		--unidepth | --robot-unidepth) new="--depth unidepth and services.unidepth" ;;
		--python | --robocasa-python | --flywheel-python | --viser-python | --xpolicy-python | --serve-python) new="python.<venv> in the deployment config" ;;
		--services) new="services_dir in the deployment config (or PI_EMBODIED_SERVICES)" ;;
		--cuda-device | --gpu-id) new="cuda_device in the deployment config (or PI_EMBODIED_CUDA_DEVICE)" ;;
		--out | --output-dir | --memory-dir | --log-dir | --serve-log-dir | --video-dir | --flywheel-root | --flash-plans | --api-slots)
			new="dirs.<kind> in the deployment config (docs/flags-migration.md)" ;;
		--ffmpeg) new="ffmpeg in the deployment config" ;;
		--robot-ros-setup) new="ros_setup in the deployment config" ;;
		*) continue ;;
		esac
		echo "$name is gone: use $new" >&2
		return 2
	done
	return 0
}
