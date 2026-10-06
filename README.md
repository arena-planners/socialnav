# socialnav

[SocialNav](https://amap-eai.github.io/SocialNav/) (Chen et al., CVPR 2026 oral, arXiv 2511.21135) behind the
arena_planners bridge: a Qwen-VL brain whose last-layer features condition a flow-matching action expert. The two
released SAFE-GRPO checkpoints carry a Qwen2-VL 2B and a Qwen2.5-VL 3B brain. The planner drives from the front camera
toward a goal given as a 2D pose. Each tick it feeds the last five frames, the robot positions at those frames and the
goal, and returns a 5-point waypoint path at 1 Hz, the frame spacing of the CityWalker data the model was trained on.

```sh
arena feature planners add socialnav
arena launch sim:=gazebo robot:=jackal[camera=vln_navid] robot.planner:=socialnav
```

`robot.planner:=socialnav` selects `robot.mobile:=vla`. The robot needs an `image` sensor in `model_params.yaml`, the
planner accepts the goal pose only (`goal_inputs: [pose]`). The two checkpoints take 4.9 GB and 8.2 GB on disk.

## Inputs

| Datasource | Topic (jackal `camera=vln_navid`) | Model input |
|---|---|---|
| `camera` (`image` sensor) | `vln_camera/image` | the last five 640x480 RGB frames, one per tick, the current frame last, each tokenized by the Qwen2-VL image processor (34x46 patches, 391 tokens) |
| `robot_pose` | tf | the five positions at those frames and the goal, in the current robot frame with x right and y forward, divided by `step_scale_m`, injected at `<input_pos1>` to `<input_pos5>` and `<input_target>` |
| goal | the subgoal | the `<input_target>` row, clipped to `goal_clip_m` |
| `subgoal` | `plan` from the nav2 planner server, `global_costmap/costmap` | the farthest point up to 6 m along the plan whose straight chord from the robot stays clear of the costmap, used in place of the goal pose, the goal pose itself until a plan arrives |

`depends.global_plan: true` makes the adapter start a nav2 planner server (`robot.mobile.global_planner`, default
`nav2/navfn`) that replans to the goal every second, and the `SubgoalGenerator` datasource turns the plan into
`subgoal`. The model steers at that point, so it only ever sees a goal in line of sight. A goal pose without a route
left the robot stuck behind walls, see Notes.

Before five ticks have passed the oldest frame and position are repeated, so an episode starts with five equal frames
and an all-zero position history. The text around the placeholders comes from `config.prompt`, because the prompt of
the training data is not published. The training cameras (CityWalker web videos, SocNav simulation and robot
recordings) are not published either, a deployment thread on the upstream tracker (issue 10) found a 640 px long side
stable.
The manifest resizes every frame to 640x480, the native size of the jackal `vln_navid` camera (90 degrees HFOV, 0.98 m
above the floor). The `vln_rgbd` variant delivers 224x224 frames, which would be upscaled, use an RGB variant.

## Actions

Each tick the expert integrates its velocity field in 5 Euler steps from `samples` initial noises and returns one path
per noise, 5 cumulative waypoints in step units, about one step apart. The planner follows the path whose last waypoint
lies closest to the goal (see Notes) and scales it by `step_scale_m` into meters in the robot frame of the observation.
The noises come from a generator seeded at every reset, so an episode replays the same draws. The bridge follows the
path with pure pursuit (lookahead 0.5 m at 0.5 m/s) for one tick, then asks again. With the defaults a tick moves the
robot 0.5 m, one step, so the recorded history has the spacing the model expects. The model has no arrival output, so
the planner lists no signals and goto phases with `signal: arrived` refuse it.

## Config

Set in `planner.yaml > config`, override per launch with `robot.mobile.config.<key>:=<value>`.

| Key | Default | Effect |
|---|---|---|
| `device` | `cuda` | Torch device. The model runs in bfloat16. |
| `variant` | `qwen2.5-vl` | `qwen2.5-vl` (Qwen2.5-VL 3B brain, the paper's model) or `qwen2-vl` (Qwen2-VL 2B brain). |
| `step_scale_m` | `0.5` | Meters per step: scales the position history and goal into model units and the waypoints back. Keep it at `chunk.speed_mps * chunk.execute_s`. |
| `goal_clip_m` | `10.0` | Goal radius cap in meters before scaling, 20 steps at the default scale. |
| `samples` | `16` | Initial noises drawn per tick, the planner follows the path that ends closest to the goal. `1` is upstream's single draw. |
| `prompt` | see `planner.yaml` | User text of the chat turn. It must hold `<input_pos1>` to `<input_pos5>` and `<input_target>` once each, in order. |

## Weights and license

`arena feature planners add socialnav` fetches both checkpoints into `model/SocialNav-Qwen2-VL-SAFE-GRPO/` and
`model/SocialNav-Qwen2.5-VL-SAFE-GRPO/` from the arena-rosnav/socialnav mirror. The source is ModelScope
(zjugyn/SocialNav-Qwen2-VL-SAFE-GRPO revision 2c729f6, zjugyn/SocialNav-Qwen2.5-VL-SAFE-GRPO revision bf3368e), the
Hugging Face repositories upstream links are empty. The checkpoints are Apache-2.0 per their ModelScope model cards,
which permits the mirror. The Qwen2-VL brain derives from Qwen2-VL-2B-Instruct (Apache-2.0), the Qwen2.5-VL brain
from Qwen2.5-VL-3B-Instruct (Qwen Research License, non-commercial research use), so the default variant is for
research use only. The code is Apache-2.0, see LICENSE and ATTRIBUTION.md. Cite Chen et al. 2025 when using this
planner:

```bibtex
@article{chen2025socialnav,
  title={SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation},
  author={Ziyi Chen and Yingnan Guo and Zedong Chu and Minghua Luo and Yanfen Shen and Mingchao Sun and Junjun Hu and Shichao Xie and Kuan Yang and Pei Shi and Zhining Gu and Lu Liu and Honglin Han and Xiaolong Wu and Mu Xu and Yu Zhang},
  journal={arXiv preprint arXiv:2511.21135},
  year={2025}
}
```

## Notes

- Measured on an RTX PRO 6000 Blackwell with torch 2.14.1+cu130, transformers 4.57.1, five 640x480 frames and 16
  samples per tick: `qwen2-vl` 4.56 GiB allocated after load, 4.98 GiB peak, 122 to 125 ms per step after a first
  step of 0.7 s. `qwen2.5-vl` 7.76 GiB allocated after load, 12.3 GiB peak, 171 to 176 ms per step. Loading from the
  page cache took 1.0 s and 1.3 s.
- One initial noise draw moves the last waypoint by about one step: over 32 draws on a street photo the last waypoint's
  standard deviation was 0.6 to 1.0 steps sideways for `qwen2-vl` and 1.0 to 1.6 steps for `qwen2.5-vl`.
- How the draws are used decides the outcome. On the `hospital_complex` stages of hospital_1 (five start and goal
  pairs 4 to 24 m apart, 2 to 8 pedestrians, 100 s per episode, `qwen2.5-vl`, steering by the goal pose) the mean of
  16 draws and their medoid reached no goal, the robot drove into an obstacle and stayed there. One draw from the same
  noise every tick reached 1 of 5 goals and looped to one side, one fresh draw per tick reached 2, and the draw ending
  closest to the goal reached 3 in each of four runs, as many as citywalker. The two remaining stages need a route
  around walls, which no draw holds and the subgoal gives.
- The route decides whether a goal behind walls is reached. On ten start and goal pairs of hospital_1 (the five
  `hospital_complex` stages and five more from shifted seeds, 4 to 24 m apart, 2 to 8 pedestrians, 100 s and one
  episode per pair) `qwen2.5-vl` reached 5 goals steering by the subgoal and 3 steering by the goal pose.
- Both monkey patches that upstream's evaluation script installs were run on the box against the vendored code: the
  waypoints and the Qwen2.5-VL vision features differed by exactly zero. Repeating the fork's `from_pretrained`
  re-initialization block on the loaded model moved the waypoints by up to 3.2 steps, see ATTRIBUTION.md.
- `tests/test_model.py` runs both checkpoints on a synthetic street scene and needs the weights and a CUDA device, the
  other tests run on CPU, `tests/test_encoder.py` needs the tokenizer files.
- transformers prints a `torchcfm` warning at load, the package only serves training and is left out.
