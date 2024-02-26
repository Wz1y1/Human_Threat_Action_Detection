from pathlib import Path

cwd = Path().cwd()
PROJECT_PATH = Path('/'.join(cwd.parts[:cwd.parts.index('PhaseTwoDelivery') + 1]))
