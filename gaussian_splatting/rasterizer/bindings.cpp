#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize(
	torch::Tensor means2D,
	torch::Tensor depths,
	torch::Tensor radii,
	torch::Tensor conics,
	torch::Tensor colors,
	torch::Tensor opacities,
	torch::Tensor background,
	int image_width,
	int image_height
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_backward(
	torch::Tensor grad_image,
	torch::Tensor means2D,
	torch::Tensor conic,
	torch::Tensor color_opacity,
	torch::Tensor values_sorted,
	torch::Tensor tile_ranges,
	torch::Tensor final_transmittance,
	torch::Tensor n_contrib,
	torch::Tensor background,
	int image_width,
	int image_height
);

torch::Tensor eval_sh(
	torch::Tensor sh_dc,
	torch::Tensor sh_rest,
	torch::Tensor dirs,
	int degree
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> eval_sh_backward_host(
	torch::Tensor grad_rgb,
	torch::Tensor sh_dc,
	torch::Tensor sh_rest,
	torch::Tensor dirs,
	int degree
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("rasterize", &rasterize, "Gaussian Splatting Rasterizer (CUDA)",
		py::arg("means2D"),
		py::arg("depths"),
		py::arg("radii"),
		py::arg("conics"),
		py::arg("colors"),
		py::arg("opacities"),
		py::arg("background"),
		py::arg("image_width"),
		py::arg("image_height")
	);
	m.def("rasterize_backward", &rasterize_backward, "Gaussian Splatting Rasterizer backward (CUDA)",
		py::arg("grad_image"),
		py::arg("means2D"),
		py::arg("conic"),
		py::arg("color_opacity"),
		py::arg("values_sorted"),
		py::arg("tile_ranges"),
		py::arg("final_transmittance"),
		py::arg("n_contrib"),
		py::arg("background"),
		py::arg("image_width"),
		py::arg("image_height")
	);
	m.def("eval_sh", &eval_sh, "Spherical harmonics colors (CUDA)",
		py::arg("sh_dc"),
		py::arg("sh_rest"),
		py::arg("dirs"),
		py::arg("degree")
	);
	m.def("eval_sh_backward", &eval_sh_backward_host, "Spherical harmonics colors backward (CUDA)",
		py::arg("grad_rgb"),
		py::arg("sh_dc"),
		py::arg("sh_rest"),
		py::arg("dirs"),
		py::arg("degree")
	);
}
