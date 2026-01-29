import numpy as np
from pathlib import Path
import pycolmap
from PIL import Image
from dataclasses import dataclass

@dataclass
class RawImage:
	"""
	uid: Unique identifier
	R: Camera rotation matrix
	t: Camera translation vector
	FovX: Horizontal fov (radians)
	FovY: Vertical fov (radians)
	image: Loaded image (H, W, C)
	width: Image width
	height: Image height
	"""
	uid: int
	R: np.ndarray
	t: np.ndarray
	FovX: float
	FovY: float
	image: np.ndarray
	width: int
	height: int


@dataclass
class PointCloud:
	"""
	points: xyz positions (N, 3)
	colors: RGB colors (N, 3)
	"""
	points: np.ndarray
	colors: np.ndarray


def focal_to_fov(focal: float, pixels: float) -> float:
	return 2 * np.arctan(pixels / (2 * focal))


class ColmapDataset:
	def __init__(self, data_path: str):
		self.data_path: Path = Path(data_path)
		self.sparse_path: Path = self.data_path / "sparse" / "0"
		self.images_path: Path = self.data_path / "images"
		
		self.reconstruction: pycolmap.Reconstruction = pycolmap.Reconstruction(str(self.sparse_path))

		self.images: list[RawImage] = self._load_cameras()
		self.point_cloud: PointCloud = self._load_point_cloud()

	def _load_cameras(self) -> list[RawImage]:
		raw_images = []
		for image_id, image in self.reconstruction.images.items():
			cam: pycolmap.Camera = self.reconstruction.cameras[image.camera_id]

			if cam.model.name == "PINHOLE":
				fx: float = cam.focal_length_x
				fy: float = cam.focal_length_y
			else:
				raise ValueError(f"Camera type {cam.model.name} not supported.")

			pose: pycolmap.Rigid3d = image.cam_from_world()
			R: np.ndarray = pose.rotation.matrix()
			t: np.ndarray = pose.translation 

			image_path: Path = self.images_path / image.name
			image_array: np.ndarray = np.array(Image.open(image_path))

			raw_image = RawImage(
				uid=image_id,
				R=R,
				t=t,
				FovX=focal_to_fov(fx, cam.width),
				FovY=focal_to_fov(fy, cam.height),
				image=image_array,
				width=cam.width,
				height=cam.height
			)
			raw_images.append(raw_image)

		return raw_images

	def _load_point_cloud(self) -> PointCloud:
		points = []
		colors = []

		for point3D_id, point3D in self.reconstruction.points3D.items():
			points.append(point3D.xyz)
			colors.append(point3D.color / 255.0)

		if len(points) != len(colors):
			raise ValueError("Error loading data, points and colors have different lengths.")

		return PointCloud(
			points=np.array(points, dtype=np.float32),
			colors=np.array(colors, dtype=np.float32)
		)
