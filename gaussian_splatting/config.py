class TrainingParams:
	"""The constants the training loop runs on, at the values the 3DGS paper uses."""

	def __init__(self) -> None:
		"""Sets every training constant to its default."""
		self.iterations = 30_000

		self.position_lr_init = 0.00016
		self.position_lr_final = 0.0000016
		self.opacity_lr = 0.025
		self.scaling_lr = 0.005
		self.rotation_lr = 0.001
		self.sh_dc_lr = 0.0025
		self.sh_rest_lr = 0.0025 / 20

		self.lambda_dssim = 0.2
		self.sh_degree_interval = 1_000

		self.densify_from_iter = 500
		self.densify_until_iter = 15_000
		self.densify_interval = 100
		self.densify_grad_threshold = 0.0002
		self.percent_dense = 0.01
		self.opacity_reset_interval = 3_000
		self.min_opacity = 0.005
		self.max_screen_size = 20

		self.eval_interval = 5_000
