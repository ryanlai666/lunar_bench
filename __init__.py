"""Local, reproducible LunarLander algorithm benchmarks."""
import sys
from pathlib import Path
# Dependencies are isolated to this project; preserve installed NumPy/Matplotlib.
sys.path.append(str(Path(__file__).resolve().parent/'deps'))

