# RoboDojo env server

One RoboDojo task (robodojo-benchmark/RoboDojo @726e9aa) on its two ARX X5 arms, on Isaac Sim 6.1 /
Isaac Lab 3.0 with `robodojo-isaac61.patch`, served over the HTTP RPC of the other env servers
(`env_server.py`, methods in services/PROTOCOL.md); `sim.py` is the RoboDojo glue (config assembly as
RoboDojo's main.py, Kit launch, readers), `primitives.py` the `code.api` registry, `flywheel.py` the
Flywheel data rules. The pi robot is `packages/embodied/src/robots/robodojo`.

## Install

```bash
bash install_isaac61.sh services/.venv-robodojo ~/RoboDojo ~/.cache/pi-embodied/robodojo-assets   # or: services/setup.sh robodojo
```

## Tasks on Isaac Sim 6.1

54 runnable tasks: RoboDojo's 42 base tasks in five capability dimensions plus 12 `_random` variants of
the Generalization tasks (`task/RoboDojo/config/_task.yml` is the shared per-task settings file, not a
task). Each was smoked on the box (RTX 5090, driver 595.71.05), one process per task: build the env,
reset eval layout 0, render the three cameras (none blank), move the left arm 3 cm up (28.7 mm measured
in every task), read RoboDojo's success check and the ground-truth objects. 48 pass. The 6 that do not
are listed in `sim.UNSUPPORTED` with the measured reason, and the env server refuses them before
starting Isaac Sim. `step_lim` is RoboDojo's control-step budget (25 Hz), `layouts` the eval layouts of
set 0, `reset s` the first reset with Kit warm.

| task | dimension | step_lim | layouts | reset s | status |
| --- | --- | --- | --- | --- | --- |
| `arrange_largest_number` | generalization | 1050 | 30 | 44.3 | ok |
| `arrange_largest_number_random` | generalization | 1050 | 30 | 106.8 | ok |
| `fold_clothes` | generalization | 500 | 55 | - | unsupported: particle cloth removed in Isaac Sim 6 |
| `fold_clothes_random` | generalization | 500 | 30 | - | unsupported: particle cloth removed in Isaac Sim 6 |
| `hang_mugs` | generalization | 800 | 30 | 34.1 | ok |
| `hang_mugs_random` | generalization | 800 | 30 | 158.8 | ok |
| `make_toast` | generalization | 1400 | 70 | 45.2 | ok |
| `make_toast_random` | generalization | 1400 | 45 | 94.4 | ok |
| `pack_objects_into_box` | generalization | 1300 | 55 | 31.4 | ok |
| `pack_objects_into_box_random` | generalization | 1300 | 30 | 45.6 | ok |
| `pour_liquid_into_cup` | generalization | 400 | 55 | - | unsupported: liquid spills; layouts tried unstable |
| `pour_liquid_into_cup_random` | generalization | 400 | 30 | - | unsupported: liquid spills; layouts tried unstable |
| `push_T` | generalization | 600 | 55 | 19.7 | ok |
| `push_T_random` | generalization | 600 | 30 | 61.2 | ok |
| `sort_nesting_dolls_by_size` | generalization | 1050 | 30 | 49.5 | ok |
| `sort_nesting_dolls_by_size_random` | generalization | 1050 | 30 | 79.3 | ok |
| `stack_blocks` | generalization | 550 | 55 | 21.6 | ok |
| `stack_blocks_random` | generalization | 550 | 30 | 69.0 | ok |
| `stack_bowls` | generalization | 800 | 55 | 19.5 | ok |
| `stack_bowls_random` | generalization | 800 | 30 | 76.4 | ok |
| `store_laptop_and_headphones` | generalization | 800 | 55 | 31.1 | ok |
| `store_laptop_and_headphones_random` | generalization | 800 | 30 | 42.6 | ok |
| `sweep_blocks` | generalization | 1000 | 55 | 27.7 | ok |
| `sweep_blocks_random` | generalization | 1000 | 30 | 62.0 | ok |
| `cover_blocks` | memory | 800 | 55 | 34.1 | ok |
| `imitate_sorting_sequence` | memory | 1600 | 65 | 61.3 | ok |
| `match_and_pick_from_conveyor` | memory | 700 | 55 | 22.9 | ok |
| `press_by_number` | memory | 700 | 65 | 27.3 | ok |
| `swap_T` | memory | 400 | 55 | 18.7 | ok |
| `swap_blocks` | memory | 700 | 55 | 19.9 | ok |
| `build_tower` | precision | 1050 | 55 | 47.6 | ok |
| `deposit_coin` | precision | 300 | 60 | 24.2 | ok |
| `fasten_screws` | precision | 1900 | 55 | 32.3 | ok |
| `insert_key` | precision | 300 | 55 | 32.3 | ok |
| `insert_tubes` | precision | 500 | 55 | 22.2 | ok |
| `play_Xylophone` | precision | 500 | 55 | 26.6 | ok |
| `plug_in_charger` | precision | 400 | 55 | - | unsupported: SDF collider fails, GPU solver faults |
| `pour_balls_into_vase` | precision | 600 | 55 | 30.9 | ok |
| `classify_objects` | long-horizon | 1100 | 55 | 38.5 | ok |
| `fill_egg_holder` | long-horizon | 700 | 65 | 45.0 | ok |
| `fill_pen_holder` | long-horizon | 1100 | 55 | 27.5 | ok |
| `make_kong` | long-horizon | 600 | 65 | 25.3 | ok |
| `organize_table` | long-horizon | 1000 | 55 | 45.2 | ok |
| `play_stacking_toy` | long-horizon | 1200 | 55 | 25.2 | ok |
| `play_tic_tac_toe` | long-horizon | 1100 | 75 | 28.6 | ok |
| `put_bottles_into_dustbin` | long-horizon | 700 | 55 | 27.5 | ok |
| `align_blocks` | open | 200 | 55 | 24.0 | ok |
| `classify_objects_by_language` | open | 1100 | 55 | 43.1 | ok |
| `general_pickup` | open | 200 | 55 | 49.6 | ok |
| `pick_from_conveyor_by_image` | open | 700 | 55 | 42.7 | ok |
| `pour_by_language` | open | 800 | 55 | - | unsupported: liquid spills; layouts tried unstable |
| `solve_equation` | open | 300 | 55 | 34.3 | ok |
| `stack_blocks_by_language` | open | 400 | 55 | 19.6 | ok |
| `store_tools_in_toolbox` | open | 900 | 55 | 32.7 | ok |
