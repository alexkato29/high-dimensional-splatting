from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import torch
from PIL import Image


@dataclass
class Camera:
	image_id: int
	name: str
	world_to_camera: torch.Tensor  # [4, 4] float32
	fx: float
	fy: float
	cx: float
	cy: float
	width: int
	height: int
	image: torch.Tensor  # [H, W, 3] uint8.


@dataclass
class PointCloud:
	points: np.ndarray  # [N, 3] float32
	colors: np.ndarray  # [N, 3] float32 in [0, 1]


class ColmapDataset:
	"""
	Reads a COLMAP scene laid out as <root>/images and <root>/sparse/0.

	Every image is decoded once, downscaled to at most max_width (3DGS results are reported in the
	1-1.6K range) and kept on the GPU. Every test_every-th image by name is held out, the same split
	the 3DGS paper evaluates on, so our test PSNR is comparable to theirs.
	"""
	def __init__(self, data_path: str, max_width: int | None = 1600, test_every: int = 8, device: str = "cuda"):
		root = Path(data_path)
		self.reconstruction = pycolmap.Reconstruction(str(root / "sparse" / "0"))

		colmap_imgs = sorted(self.reconstruction.images.values(), key=lambda colmap_img: colmap_img.name)
		cameras = [self._load_camera(colmap_img, root / "images", max_width, device) for colmap_img in colmap_imgs]
		self.train_cameras: list[Camera] = [c for i, c in enumerate(cameras) if i % test_every != 0]
		self.test_cameras: list[Camera] = [c for i, c in enumerate(cameras) if i % test_every == 0]

		points = list(self.reconstruction.points3D.values())
		self.point_cloud = PointCloud(
			points=np.array([p.xyz for p in points], dtype=np.float32),
			colors=np.array([p.color for p in points], dtype=np.float32) / 255.0,
		)

		# Radius of the training cameras around their centroid, padded 10%. Scales the
		# position learning rate and densification thresholds to the scene's size.
		centers = torch.stack([c.world_to_camera.inverse()[:3, 3] for c in self.train_cameras])
		self.extent: float = 1.1 * (centers - centers.mean(dim=0)).norm(dim=1).max().item()

	def _load_camera(self, colmap_img: pycolmap.Image, images_dir: Path, max_width: int | None, device: str) -> Camera:
		colmap_cam = self.reconstruction.cameras[colmap_img.camera_id]
		# The rasterizer is a pure pinhole projection, so lens distortion must already be removed
		# (colmap image_undistorter). Otherwise every pixel lands slightly in the wrong place.
		if colmap_cam.model.name not in ("SIMPLE_PINHOLE", "PINHOLE"):
			raise ValueError(f"{colmap_img.name}: camera model {colmap_cam.model.name} is unsupported, undistort the scene first")

		# The image file, not COLMAP, decides the resolution. Datasets often ship images already
		# downscaled from what SfM ran on. Only the aspect ratio has to agree, since intrinsics
		# are rescaled to whatever we load. Downscaling keeps it to within rounding (well under 1%),
		# while a rotated or cropped image changes it.
		pil_img = Image.open(images_dir / colmap_img.name)
		width, height = pil_img.size
		camera_aspect_ratio = colmap_cam.width / colmap_cam.height
		image_aspect_ratio = width / height
		if abs(image_aspect_ratio / camera_aspect_ratio - 1) > 0.01:
			raise ValueError(f"{colmap_img.name} is {pil_img.size}, not the aspect ratio of its COLMAP camera {(colmap_cam.width, colmap_cam.height)}")
		if max_width is not None and width > max_width:
			width, height = max_width, round(height * max_width / width)
		sx, sy = width / colmap_cam.width, height / colmap_cam.height

		# JPEGs can decode directly at 1/2, 1/4 or 1/8 scale in the DCT domain. Asking for the
		# smallest one that is still >= the target skips most of the decode before the real resize.
		pil_img.draft("RGB", (width, height))
		pil_img = pil_img.convert("RGB").resize((width, height))

		world_to_camera = torch.eye(4)
		world_to_camera[:3] = torch.from_numpy(colmap_img.cam_from_world().matrix())

		return Camera(
			image_id=colmap_img.image_id,
			name=colmap_img.name,
			world_to_camera=world_to_camera.to(device),
			# COLMAP and the rasterizer both put pixel centers at +0.5, so rescaling is a plain multiply.
			fx=colmap_cam.focal_length_x * sx,
			fy=colmap_cam.focal_length_y * sy,
			cx=colmap_cam.principal_point_x * sx,
			cy=colmap_cam.principal_point_y * sy,
			width=width,
			height=height,
			image=torch.from_numpy(np.array(pil_img)).to(device),
		)
