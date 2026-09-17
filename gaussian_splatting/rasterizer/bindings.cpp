#include "api.h"


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
	m.def("project_gaussians", &project_gaussians, "Project gaussians into the image plane (CUDA)");
	m.def("project_gaussians_backward", &project_gaussians_backward, "Projection backward (CUDA)");
}
