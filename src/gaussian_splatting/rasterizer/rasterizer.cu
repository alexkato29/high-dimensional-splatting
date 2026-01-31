#include <torch/extensions.h>
#include <cuda_runtime.h>
#include <helpers.h>


__global__ void project_gaussians(int num_gaussians) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > num_gaussians) return;
}

torch::Tensor rasterize(
	torch::Tensor means3D,
	torch::Tensor scales,
	torch::Tensor rotations,
	torch::Tensor opacities,
	torch::Tensor colors,
	torch::Tensor viewmatrix,
	torch::Tensor projmatrix,
	int image_height,
	int image_width,
	float tanfovx,
	float tanfovy
) {
	// Torch handles the allocation of its tensor to GPU memory, don't need to use cudaMemCpy myself
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	torch::Tensor output = torch::zeros({image_height, image_width, 3}, options);

	int num_gaussians = means3D.size(0);

	float* means3D_ptr = means3D.data_ptr<float>();
	float* scales_ptr = scales.data_ptr<float>();
	float* rotations_ptr = rotations.data_ptr<float>();
	float* opacities_ptr = opacities.data_ptr<float>();
	float* colors_ptr = colors.data_ptr<float>();
	float* viewmatrix_ptr = viewmatrix.data_ptr<float>();
	float* projmatrix_ptr = projmatrix.data_ptr<float>();

	int threads = 256;
	int blocks = (num_gaussians + threads - 1) / threads;
	project_gaussians<<<blocks, threads>>>(
		// Params TBD
	);

	// Step 2: sort gaussians per tile
	// Step 3: alpha blend
	dim3 block(16, 16);
	dim3 grid(
		(image_width + block.x - 1) / block.x,
		(image_height + block.y - 1) / block.y 
	);

	return output;
}