class TrainingParams:
	def __init__(self):
		self.iterations: int = 30_000

		self.position_lr_init: float = 0.00016
		self.position_lr_final: float = 0.0000016
		self.opacity_lr: float = 0.025
		self.scaling_lr: float = 0.005
		self.rotation_lr: float = 0.001
		self.sh_dc_lr: float = 0.0025
		self.sh_rest_lr: float = 0.0025 / 20

		self.lambda_dssim: float = 0.2
		self.sh_degree_interval: int = 1_000

		self.densify_from_iter: int = 500
		self.densify_until_iter: int = 15_000
		self.densify_interval: int = 100
		self.densify_grad_threshold: float = 0.0002
		self.percent_dense: float = 0.01
		self.opacity_reset_interval: int = 3_000
		self.min_opacity: float = 0.005
		self.max_screen_size: int = 20

		self.eval_interval: int = 5_000
