import os

from hypothesis import settings

settings.register_profile("dev", max_examples=20, deadline=None)
settings.register_profile("full", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
