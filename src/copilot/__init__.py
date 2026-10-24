"""Responsible Weather Incident Copilot.

Decision support for a city operations center. The package never sends alerts or
dispatches crews on its own: notification sits behind an explicit human approval.
"""

import warnings

# LangGraph's checkpoint serializer warns about a default that changes in a future release.
# We only checkpoint plain JSON-compatible state, so the warning is not actionable here.
warnings.filterwarnings("ignore", message=r"The default value of `allowed_objects`")

__version__ = "0.1.0"
