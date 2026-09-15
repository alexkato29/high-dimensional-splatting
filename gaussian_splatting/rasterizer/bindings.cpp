#include <torch/extension.h>

torch::Tensor rasterize(
	torch::Tensor means2D,
	torch::Tensor depths,
	torch::Tensor radii,
	torch::Tensor conics,
	torch::Tensor colors,
	torch::Tensor opacities,
	int image_width,
	int image_height
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("rasterize", &rasterize, "Gaussian Splatting Rasterizer (CUDA)",
		py::arg("means2D"),
		py::arg("depths"),
		py::arg("radii"),
		py::arg("conics"),
		py::arg("colors"),
		py::arg("opacities"),
		py::arg("image_width"),
		py::arg("image_height")
	);
}
