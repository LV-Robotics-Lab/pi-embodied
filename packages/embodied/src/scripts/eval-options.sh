#!/usr/bin/env bash
# Shared evaluation metadata for pi flags. Source from a robot's eval.sh.
# eval_options_defaults [default-turns] sets the ordinary defaults. A robot may
# override its own defaults before eval_parse_options "$@". Unknown arguments
# go to an optional eval_robot_option <argument> <next-value> callback and remain
# untouched in the original argv passed to pi. Normalize once after parsing.
# Task selection, scoring, protocol timeouts and result schemas stay with robots.

eval_options_defaults() {
	model="" thinking="" turns=${1:-0} limit=${TIME_LIMIT:-1800} limited="" units=false stateless=false
	anchor=false
	approval=off max_tool_calls=0 max_tokens=0
	vdm=false vdm_model="" vdm_wrist=false vdm_video=false vdm_video_frames=8
	privileged=false
	fallback_model="" fallback_after=2 fallback_retry=0
	code=false code_api="" code_oracle="" code_timeout="" code_max_calls="" code_max_move="" code_helpers=false
	units_opts="" ft_opts=""
	unset units_plugins
	extras="" eval_extras=()
	return 0
}

eval_parse_options() {
	local args=("$@")
	local i a
	for ((i = 0; i < ${#args[@]}; i++)); do
		case ${args[i]} in
		--model) model=${args[i + 1]:-} ;;
		--thinking) thinking=${args[i + 1]:-} ;;
		--model=*) model=${args[i]#*=} ;;
		--thinking=*) thinking=${args[i]#*=} ;;
		--max-turns) turns=${args[i + 1]:-0} ;;
		--max-turns=*) turns=${args[i]#*=} ;;
		--time-limit) limit=${args[i + 1]:-0} limited=1 ;;
		--time-limit=*) limit=${args[i]#*=} limited=1 ;;
		--units) [[ ${args[i + 1]:---} == --* ]] && units=true || units=${args[i + 1]} ;;
		--units=*) units=${args[i]#*=} ;;
		--units-plugins) units_plugins=${args[i + 1]-} ;;
		--units-plugins=*) units_plugins=${args[i]#*=} ;;
		# The units' experiment knobs (stage cap, action ablation, point self-check) are part of the units mode too.
		--units-stage-steps | --units-ablation | --units-point-verify) units_opts+="+${args[i]#--units-}=${args[i + 1]-}" ;;
		--units-stage-steps=* | --units-ablation=* | --units-point-verify=*) units_opts+="+${args[i]#--units-}" ;;
		# pi sets a boolean flag to true whatever value it is given (`--stateless=false` runs stateless)
		# and takes a following word as that value: only the forms that say what pi runs are accepted.
		--stateless) case ${args[i + 1]:-} in "" | -* | @* | true) stateless=true ;; *)
			echo "--stateless takes no value: pi would run stateless and swallow '${args[i + 1]}'" >&2 && exit 2 ;;
		esac ;;
		--stateless=true) stateless=true ;;
		--stateless=*)
			echo "${args[i]}: pi ignores a boolean flag's value and would run stateless; omit --stateless for a stateful run" >&2
			exit 2
			;;
		# --privileged (simulator ground truth, ground_truth_poses) is a boolean like --stateless.
		--privileged) case ${args[i + 1]:-} in "" | -* | @* | true) privileged=true ;; *)
			echo "--privileged takes no value: pi would turn it on and swallow '${args[i + 1]}'" >&2 && exit 2 ;;
		esac ;;
		--privileged=true) privileged=true ;;
		--privileged=*)
			echo "${args[i]}: pi ignores a boolean flag's value and would turn it on; omit --privileged for a run without ground truth" >&2
			exit 2
			;;
		--vdm | --vdm-wrist | --vdm-video) case ${args[i + 1]:-} in "" | -* | @* | true) case ${args[i]} in --vdm) vdm=true ;; --vdm-wrist) vdm_wrist=true ;; *) vdm_video=true ;; esac ;; *)
			echo "${args[i]} takes no value: pi would turn it on and swallow '${args[i + 1]}'" >&2 && exit 2 ;;
		esac ;;
		--vdm=true) vdm=true ;;
		--vdm-wrist=true) vdm_wrist=true ;;
		--vdm-video=true) vdm_video=true ;;
		--vdm-video-frames) vdm_video_frames=${args[i + 1]:-8} ;;
		--vdm-video-frames=*) vdm_video_frames=${args[i]#*=} ;;
		--vdm=* | --vdm-wrist=* | --vdm-video=*)
			echo "${args[i]}: pi ignores a boolean flag's value and would turn it on; omit it to leave it off" >&2
			exit 2
			;;
		--vdm-model) vdm_model=${args[i + 1]:-} ;;
		--vdm-model=*) vdm_model=${args[i]#*=} ;;
		--fallback-model) fallback_model=${args[i + 1]:-} ;;
		--fallback-model=*) fallback_model=${args[i]#*=} ;;
		--fallback-after) fallback_after=${args[i + 1]:-2} ;;
		--fallback-after=*) fallback_after=${args[i]#*=} ;;
		--fallback-retry-primary) fallback_retry=${args[i + 1]:-0} ;;
		--fallback-retry-primary=*) fallback_retry=${args[i]#*=} ;;
		# --approval (motion approval, src/capabilities/operator.ts) and the --max-tool-calls / --max-tokens budgets (src/robot.ts).
		--approval) approval=${args[i + 1]:-off} ;;
		--approval=*) approval=${args[i]#*=} ;;
		--max-tool-calls) max_tool_calls=${args[i + 1]:-0} ;;
		--max-tool-calls=*) max_tool_calls=${args[i]#*=} ;;
		--max-tokens) max_tokens=${args[i + 1]:-0} ;;
		--max-tokens=*) max_tokens=${args[i]#*=} ;;
		# --code / --code-api / --code-oracle (run_code, packages/embodied/src/modes/code) are string flags like --units.
		--code) [[ ${args[i + 1]:---} == --* ]] && code=true || code=${args[i + 1]} ;;
		--code=*) code=${args[i]#*=} ;;
		--code-api) code_api=${args[i + 1]:-high} ;;
		--code-api=*) code_api=${args[i]#*=} ;;
		--code-oracle) code_oracle=${args[i + 1]:-} ;;
		--code-oracle=*) code_oracle=${args[i]#*=} ;;
		# The code budget is part of the code mode's configuration (result.json's code_budget_flags).
		--code-timeout) code_timeout=${args[i + 1]-} ;;
		--code-timeout=*) code_timeout=${args[i]#*=} ;;
		--code-max-calls) code_max_calls=${args[i + 1]-} ;;
		--code-max-calls=*) code_max_calls=${args[i]#*=} ;;
		--code-max-move) code_max_move=${args[i + 1]-} ;;
		--code-max-move=*) code_max_move=${args[i]#*=} ;;
		--code-helpers | --code-helpers=true) code_helpers=true ;;
		# --anchor-image (keep the first camera frame in context) is a boolean like --stateless.
		--anchor-image) case ${args[i + 1]:-} in "" | -* | @* | true) anchor=true ;; *)
			echo "--anchor-image takes no value: pi would turn it on and swallow '${args[i + 1]}'" >&2 && exit 2 ;;
		esac ;;
		--anchor-image=true) anchor=true ;;
		--anchor-image=*)
			echo "${args[i]}: pi ignores a boolean flag's value and would turn it on; omit --anchor-image to leave it off" >&2
			exit 2
			;;
		# The OpenETA extras (src/primitives/optional.ts, capabilities/objects.ts, capabilities/web.ts) are booleans like --stateless;
		# robot.ts records the ones that were on as `extras`, and a result with other extras is another configuration.
		--waypoints | --align-wrist | --grasp-advisor | --object-memory | --web-tools) case ${args[i + 1]:-} in "" | -* | @* | true) eval_extras+=("${args[i]#--}") ;; *)
			echo "${args[i]} takes no value: pi would turn it on and swallow '${args[i + 1]}'" >&2 && exit 2 ;;
		esac ;;
		--waypoints=true | --align-wrist=true | --grasp-advisor=true | --object-memory=true | --web-tools=true) a=${args[i]#--} && eval_extras+=("${a%=true}") ;;
		--waypoints=* | --align-wrist=* | --grasp-advisor=* | --object-memory=* | --web-tools=*)
			echo "${args[i]}: pi ignores a boolean flag's value and would turn it on; omit it to leave it off" >&2
			exit 2
			;;
		# The fine-tuned policy's flags (src/modes/finetuned) are its configuration, recorded as result.json's
		# ft_flags in the order given; --ft-endpoint and --ft-api-key only say where the adapter is served.
		--ft-endpoint | --ft-api-key | --ft-endpoint=* | --ft-api-key=*) ;;
		--ft-*=*) ft_opts+="+${args[i]#--ft-}" ;;
		--ft-*) ft_opts+="+${args[i]#--ft-}=${args[i + 1]-}" ;;
		*) if declare -F eval_robot_option >/dev/null; then eval_robot_option "${args[i]}" "${args[i + 1]-}"; fi ;;
		esac
	done
	return 0
}

eval_normalize_options() {
	[ "$units" = pure ] && units=true
	[ "$code" = pure ] && code=true
	[ "$units" != false ] && [ -n "${units_plugins+x}" ] && units="$units+plugins=$units_plugins"
	[ "$units" != false ] && units="$units${units_opts-}"
	[ "$vdm_video" = true ] && vdm_video=$vdm_video_frames || vdm_video=""
	extras=$(printf '%s\n' ${eval_extras[@]+"${eval_extras[@]}"} | sort -u | paste -sd, -)
	# Only a finetuned/<adapter> model reads the --ft-* flags: "default" with none, else the flags as given.
	case $model in
	finetuned/*) export FT_FLAGS="${ft_opts:-+default}" && FT_FLAGS=${FT_FLAGS#+} ;;
	*) export FT_FLAGS="" ;;
	esac
	export CODE_BUDGET_FLAGS="timeout=$code_timeout+max_calls=$code_max_calls+max_move=$code_max_move+helpers=$code_helpers${code_oracle:++oracle}"
	return 0
}
