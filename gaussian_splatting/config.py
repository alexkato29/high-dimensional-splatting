class TrainingParams:
	def __init__(self):
		self.iterations: int = 30_000
		self.position_lr: float = 0.00016
		self.opacity_lr: float = 0.025
		self.scaling_lr: float = 0.005
		self.rotation_lr: float = 0.001
		self.rgb_lr: float = 0.0025
