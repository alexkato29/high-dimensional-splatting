#include <torch/extension.h>

torch::Tensor rasterize(
	torch::Tensor means3D,
	torch::Tensor scales,
	torch::Tensor quaternions,
	torch::Tensor opacities,
	torch::Tensor colors,
	torch::Tensor world_to_cam_matrix,
	float focal_x,
	float focal_y,
	float c_x,
	float c_y,
	int image_width,
	int image_height
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("rasterize", &rasterize, "Gaussian Splatting Rasterizer (CUDA)",
		py::arg("means3D"),
		py::arg("scales"),
		py::arg("quaternions"),
		py::arg("opacities"),
		py::arg("colors"),
		py::arg("world_to_cam_matrix"),
		py::arg("focal_x"),
		py::arg("focal_y"),
		py::arg("c_x"),
		py::arg("c_y"),
		py::arg("image_width"),
		py::arg("image_height")
	);
}
