#pragma once
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize_features(
	torch::Tensor features,
	torch::Tensor means2D,
	torch::Tensor conic,
	torch::Tensor color_opacity,
	torch::Tensor values_sorted,
	torch::Tensor tile_ranges,
	int image_width,
	int image_height
);

torch::Tensor eval_sh(
	torch::Tensor sh_dc,
	torch::Tensor sh_rest,
	torch::Tensor dirs,
	int degree,
	torch::Tensor radii
);

torch::Tensor eval_sh_backward_adam_host(
	torch::Tensor grad_rgb,
	torch::Tensor sh_dc,
	torch::Tensor sh_rest,
	torch::Tensor dirs,
	int degree,
	torch::Tensor radii,
	torch::Tensor m_dc,
	torch::Tensor v_dc,
	torch::Tensor m_rest,
	torch::Tensor v_rest,
	torch::Tensor step_dc,
	torch::Tensor step_rest,
	double lr_dc,
	double lr_rest,
	double beta1,
	double beta2,
	double eps
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> project_gaussians(
	torch::Tensor means,
	torch::Tensor scales,
	torch::Tensor quats,
	torch::Tensor opacities,
	torch::Tensor world_to_camera,
	float fx, float fy, float cx, float cy,
	int width, int height,
	float near_plane, float low_pass, float fov_scale
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> ssim_forward(
	torch::Tensor rendered,
	torch::Tensor gt
);

torch::Tensor ssim_backward(
	torch::Tensor grad_mean,
	torch::Tensor rendered,
	torch::Tensor gt,
	torch::Tensor d_mu,
	torch::Tensor d_aa,
	torch::Tensor d_ab
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> project_gaussians_backward(
	torch::Tensor grad_means2D,
	torch::Tensor grad_conics,
	torch::Tensor means,
	torch::Tensor scales,
	torch::Tensor quats,
	torch::Tensor world_to_camera,
	float fx, float fy, float cx, float cy,
	int width, int height,
	float near_plane, float low_pass, float fov_scale
);
