from os import getenv
from os.path import join
from pathlib import Path

DATADIR = getenv("METRICS_HOME", join(getenv("XDG_CACHE_HOME", join(Path.home(), ".cache")), "metrics"))
Path(DATADIR).mkdir(parents=True, exist_ok=True)
