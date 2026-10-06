from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import planner

_FILES = ("tokenizer_config.json", "tokenizer.json", "added_tokens.json", "preprocessor_config.json")


@pytest.fixture(scope="module", params=sorted(planner.VARIANTS))
def encoder(request: pytest.FixtureRequest) -> planner.Encoder:
    variant = request.param
    missing = [name for name in _FILES if not (planner.model_dir(variant) / name).is_file()]
    if missing:
        pytest.skip(f"{variant} tokenizer files {missing} missing, run `arena feature planners add socialnav`")
    return planner.Encoder(planner.load_processor(variant), planner.DEFAULT_PROMPT)


def test_waypoint_token_ids_are_contiguous(encoder: planner.Encoder):
    ids = [encoder.token_ids[token] for token in planner.WAYPOINT_TOKENS]
    assert ids == list(range(ids[0], ids[0] + 6))
    assert ids[0] == encoder.processor.tokenizer.convert_tokens_to_ids("<input_pos1>")


def test_text_follows_upstream_chat_template(encoder: planner.Encoder):
    text = encoder.text
    assert text.startswith("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n")
    assert text.count("<|vision_start|><|image_pad|><|vision_end|>") == 5
    assert text.endswith(planner.DEFAULT_PROMPT + "<|im_end|>\n<|im_start|>assistant\n<|im_end|>")


def test_batch_holds_five_frames_and_the_placeholders_in_order(encoder: planner.Encoder, frame: np.ndarray):
    batch = encoder.batch([Image.fromarray(frame)] * 5)
    assert batch["image_grid_thw"].shape == (5, 3)
    assert batch["image_grid_thw"][0].tolist() == [1, 34, 46]
    ids = batch["input_ids"][0].tolist()
    found = [ids.index(encoder.token_ids[token]) for token in planner.WAYPOINT_TOKENS]
    assert found == sorted(found)
    assert sum(ids.count(encoder.token_ids[token]) for token in planner.WAYPOINT_TOKENS) == 6
    image_token = encoder.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    assert ids.count(image_token) == 5 * 34 * 46 // 4
    assert batch["attention_mask"].all()


def test_batch_refuses_a_wrong_frame_count(encoder: planner.Encoder, frame: np.ndarray):
    with pytest.raises(ValueError):
        encoder.batch([Image.fromarray(frame)] * 4)
