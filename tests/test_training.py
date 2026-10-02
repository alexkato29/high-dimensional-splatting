import argparse
import os
from pathlib import Path

import pytest
import torch

from gaussian_splatting.main import evaluate, train
from gaussian_splatting.utils.dataset import ColmapDataset
from gaussian_splatting.utils.gaussian import GaussianModel

SCENE = os.environ.get("GS_SCENE")
ITERATIONS = 1000
# The Truck scene from Tanks & Temples (what I've been using) sees ~8.5dB gain after 1000 iterations, on average.
# When opacities or scales are frozen, the gain falls to ~6.5dB, on average. This sits between the correct and
# incorrect gain bounds.
MIN_GAIN_DB = 7.5

pytestmark = [
	pytest.mark.skipif(not torch.cuda.is_available(), reason="training needs CUDA"),
	pytest.mark.skipif(SCENE is None, reason="set GS_SCENE to a COLMAP scene to run the training smoke test"),
]


def test_training_improves_the_scene(tmp_path):
	assert Path(SCENE).resolve().parts[-2:] == ("tandt", "truck"), (
		f"MIN_GAIN_DB is calibrated for tandt/truck, recalibrate it before training on {SCENE}")

	# 1000 iterations cover densification (600 to 1000), the first SH degree step (1000) and one eval
	# plus checkpoint at the end.
	args = argparse.Namespace(data_path=SCENE, iterations=ITERATIONS, eval_interval=ITERATIONS, max_width=1600,
							  test_every=8, output=tmp_path, seed=0)
	train(args)
	state = torch.load(tmp_path / "model.pt")

	dataset = ColmapDataset(SCENE)
	with torch.no_grad():
		untrained_psnr = evaluate(dataset, GaussianModel(dataset.point_cloud))

	initial_count = len(dataset.point_cloud.points)
	final_count = state["params"]["means"].shape[0]
	print(f"untrained {untrained_psnr:.2f} dB, trained {state['test_psnr']:.2f} dB, "
		  f"gaussians {initial_count} -> {final_count}")

	assert state["iteration"] == ITERATIONS
	assert state["test_psnr"] > untrained_psnr + MIN_GAIN_DB
	assert final_count > initial_count, "densification never added a gaussian"
	for name, param in state["params"].items():
		assert param.isfinite().all(), f"{name} has non-finite values"
