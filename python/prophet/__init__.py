# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from prophet.forecaster import Prophet
from pathlib import Path
import re

# Get the directory of the current file
here = Path(__file__).parent.resolve()

# Path to the version file
version_file = here / "__version__.py"

# Read the version file and extract __version__
with open(version_file, "r") as f:
    for line in f:
        match = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', line)
        if match:
            __version__ = match.group(1)
            break
    else:
        raise ValueError("Could not find __version__ in __version__.py")
