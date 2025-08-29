# extract test indices used

import os
import sys
import argparse
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src import data_utils as du

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_name", help="basename (no .nc) of your dataset")
parser.add_argument(
    "-t",
    "--test_size",
    type=float,
    default=0.2,
    help="testing set proportion, must be between 0 and 1, default is 0.2",
)
args = parser.parse_args()

params = {
    "data_src": args.data_name,
    "random_seed": 1952,
}

# Open dataset
outputs = du.open_mass_dataset(
    name=params["data_src"],
    data_dir=Path(__file__).parent.parent.parent / "data",
    test_size=args.test_size,
    random_state=params["random_seed"],
)
print(*outputs["idx_test"])
