import sys
from unittest.mock import MagicMock

pycolmap_mock = MagicMock()
sys.modules['pycolmap'] = pycolmap_mock
