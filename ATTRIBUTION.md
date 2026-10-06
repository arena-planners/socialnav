# Attribution

## Code

Model code adapted from AMAP-EAI/SocialNav (commit b958f06, 2026-08-11).
Source: https://github.com/AMAP-EAI/SocialNav

`socialnav_model/` holds these files of the upstream `transformers/` tree (a patched transformers 4.57.1) verbatim,
except that the relative `...` imports became absolute `transformers.` imports resolved by stock transformers 4.57.1:

- `transformers/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py`
- `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`

Each is the stock 4.57.1 file plus the SocialNav action expert: `MultiLayerEmbedding`, `ActionFormerHead`,
`TransformerForDiffusion`, `FlowMatchingActionHead`, and the expert constructor, `_encode_input_waypoints`,
`_pool_action_feature`, `recover_waypoints_from_pred` and the `train_branch="fm"` branch of `forward` on the
`ForConditionalGeneration` class. The `torchcfm` import stays optional as upstream wrote it, the package is not
installed, and the fallback it guards only affects training.

Not vendored, with the reason:

- `src/train/monkey_patch_forward.py` and `src/train/monkey_patch_vision.py`: upstream's evaluation script installs
  them, the planner does not. The forward patch only differs from the stock model forward for batches without images
  and for cached decoding, the vision patch only in window padding arithmetic and gradient checkpointing.
  `tests/test_model.py` does not cover this, a one-off run of both patches against the vendored code on the GPU box
  gave waypoints and vision features with zero difference for both checkpoints, see the Notes in README.md.
- `src/train/sde_with_logprob.py` (SAFE-GRPO sampling): imported lazily inside `_grpo_sample_actions`, which only runs
  with `grpo_mode=True`. The planner never sets it.
- The fork's `modeling_utils.py`: its `from_pretrained` re-initializes `query_action`, the ActionFormer and the
  velocity predictor's positional embeddings after the checkpoint is loaded, which replaces the trained action expert
  with random weights (upstream issue #10). Stock transformers keeps the loaded weights, and `Policy` refuses a
  checkpoint with missing keys.
- The Qwen3-VL modeling files: no released checkpoint uses them, and the two vendored files never import them.

`planner.py` ports `SocialNavModel.__init__`, `_resolve_special_tokens` and `infer_one` of `utils/citywalker.py`: the
same expert kwargs, chat template (`add_generation_prompt=True` plus a closing `<|im_end|>`), processor call and
forward call. The planner assembles `input_waypoints` the way CityWalker's dataset builds its `input_target`
embedding (ai4ce/CityWalker `data/citywalk_dataset.py`): the five positions of the last five frames and the goal,
expressed in the current pose frame with x right and y forward, divided by the step scale. The step scale is a config
constant instead of the per-trajectory mean step length and the goal radius is clipped. Upstream integrates one flow
from unseeded noise. The planner integrates `samples` flows from the noises of a generator seeded at every reset,
passed through `select_noise`, and follows the one that ends closest to the goal. The prompt text of the
training data is not published, so the planner's prompt is a config value that keeps the six placeholder tokens.

## Weights

ModelScope zjugyn/SocialNav-Qwen2-VL-SAFE-GRPO (revision 2c729f6) and
zjugyn/SocialNav-Qwen2.5-VL-SAFE-GRPO (revision bf3368e), both with `license: Apache License 2.0` in their ModelScope
model cards and no further terms. The Hugging Face repositories named in the upstream README hold only a
`.gitattributes` file. Apache-2.0 permits redistribution, so the inference files (safetensors, tokenizer, processor
and model configs, without the optimizer and trainer state) are mirrored at arena-rosnav/socialnav. The brains
derive from Qwen/Qwen2-VL-2B-Instruct (Apache-2.0) and Qwen/Qwen2.5-VL-3B-Instruct (Qwen Research License
Agreement, non-commercial research use), which apply to the respective variant as well.

Runtime dependency: stock transformers 4.57.1 from PyPI, the version the upstream fork is based on.

See LICENSE for upstream terms (Apache-2.0).

Cite:

```bibtex
@article{chen2025socialnav,
  title={SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation},
  author={Ziyi Chen and Yingnan Guo and Zedong Chu and Minghua Luo and Yanfen Shen and Mingchao Sun and Junjun Hu and Shichao Xie and Kuan Yang and Pei Shi and Zhining Gu and Lu Liu and Honglin Han and Xiaolong Wu and Mu Xu and Yu Zhang},
  journal={arXiv preprint arXiv:2511.21135},
  year={2025}
}
```
