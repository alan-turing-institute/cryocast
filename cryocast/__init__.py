import logging
import os
import sys
import warnings

logging.getLogger("anemoi").setLevel(logging.WARNING)
logging.getLogger("argopy").setLevel(logging.WARNING)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if "torch" in sys.modules:
    warnings.warn(
        "PYTORCH_ENABLE_MPS_FALLBACK was set after torch was imported. "
        "This means that it might not have any effect. "
        "If you have problems, please try setting this environment variable before "
        "importing torch.",
        stacklevel=2,
    )
