import os

import numpy as np
import pytest
import torch

from gaussian_splatting.utils.dataset import Camera, ColmapDataset

SCENE = os.environ.get("GS_SCENE", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pytestmark = pytest.mark.skipif(not SCENE, reason="set GS_SCENE to a COLMAP scene directory")


@pytest.fixture(scope="module")
def dataset() -> ColmapDataset:
	return ColmapDataset(SCENE, device=DEVICE)


def cameras(ds: ColmapDataset) -> list[Camera]:
	return ds.train_cameras + ds.test_cameras


def observations(ds: ColmapDataset, cam: Camera) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Collects the 3D points COLMAP triangulated from one image, with where COLMAP projects them.

	Args:
		ds: Dataset holding the COLMAP reconstruction.
		cam: Camera whose observations to collect.

	Returns:
		A tuple of three float64 tensors on the CPU.
			xyz: [M, 3] world positions of the observed points.
			expected: [M, 2] COLMAP's own projection of each point, rescaled to the loaded image size.
			colors: [M, 3] COLMAP point colors in [0, 1].
	"""
	colmap_img = ds.reconstruction.images[cam.image_id]
	colmap_cam = ds.reconstruction.cameras[colmap_img.camera_id]
	observed = [p for p in colmap_img.points2D if p.has_point3D()]
	points = [ds.reconstruction.points3D[p.point3D_id] for p in observed]
	xyz_np = np.array([p.xyz for p in points])
	# COLMAP pixel coordinates start at the image corner, so a resize scales them about the origin.
	scale = torch.tensor([cam.width / colmap_cam.width, cam.height / colmap_cam.height], dtype=torch.float64)
	expected = torch.from_numpy(colmap_cam.img_from_cam(colmap_img.cam_from_world() * xyz_np)) * scale
	colors = torch.tensor([p.color for p in points], dtype=torch.float64) / 255
	return torch.from_numpy(xyz_np), expected, colors


def project(cam: Camera, xyz: torch.Tensor) -> torch.Tensor:
	"""Pinhole projection of world points into a camera's pixels.

	Args:
		cam: Camera to project into.
		xyz: [M, 3] float64 world positions.

	Returns:
		[M, 2] float64 pixel coordinates, origin at the image corner.
	"""
	world_to_camera = cam.world_to_camera.cpu().double()
	p = xyz @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
	return torch.stack([cam.fx * p[:, 0] / p[:, 2] + cam.cx, cam.fy * p[:, 1] / p[:, 2] + cam.cy], dim=-1)


def test_points_project_where_colmap_projects_them(dataset: ColmapDataset) -> None:
	for cam in cameras(dataset):
		xyz, expected, _ = observations(dataset, cam)
		assert (project(cam, xyz) - expected).abs().max() < 1e-2, cam.name


def test_center_is_the_camera_origin(dataset: ColmapDataset) -> None:
	for cam in cameras(dataset):
		world_to_camera = cam.world_to_camera.cpu().double()
		origin_world = -world_to_camera[:3, :3].T @ world_to_camera[:3, 3]
		tol = 1.0e-5 * torch.norm(world_to_camera[:3, 3])
		assert (cam.center.cpu() - origin_world).abs().max() <= tol, cam.name


def test_point_colors_match_image_pixels(dataset: ColmapDataset) -> None:
	# Lighting, view dependence, and JPEG noise set an error floor no fixed tolerance can know.
	# So, the loaded image must beat broken copies of itself pooled over every camera so a
	# single, near-gray image cannot hide or fake a channel swap.
	errors = {"loaded": [], "flipped": [], "channels_reversed": []}
	for cam in cameras(dataset):
		xyz, _, colors = observations(dataset, cam)
		projected = project(cam, xyz)
		u = projected[:, 0].floor().long().clamp(0, cam.width - 1)
		v = projected[:, 1].floor().long().clamp(0, cam.height - 1)
		image = cam.image.cpu().double() / 255
		for name, candidate in (("loaded", image), ("flipped", image.flip(0)), ("channels_reversed", image[..., [2, 1, 0]])):
			errors[name].append((candidate[v, u] - colors).abs().mean(dim=-1))
	median = {name: torch.cat(e).median() for name, e in errors.items()}
	assert median["loaded"] < 0.5 * median["flipped"], median
	assert median["loaded"] < 0.5 * median["channels_reversed"], median


def test_projections_scale_with_max_width(dataset: ColmapDataset) -> None:
	half = ColmapDataset(SCENE, max_width=dataset.train_cameras[0].width // 2, device=DEVICE)
	for full_cam, half_cam in zip(cameras(dataset), cameras(half)):
		xyz, _, _ = observations(dataset, full_cam)
		# The rounded height makes x and y scale by slightly different factors.
		ratio = torch.tensor([half_cam.width / full_cam.width, half_cam.height / full_cam.height], dtype=torch.float64)
		assert (project(half_cam, xyz) - project(full_cam, xyz) * ratio).abs().max() < 1e-2, full_cam.name
