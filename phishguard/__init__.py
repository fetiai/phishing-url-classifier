"""phishguard -- phishing URL classification library.

Module boundary rules, stated as rules because they are load-bearing:

1. ``phishguard`` never imports ``streamlit``. ``app/`` never imports ``sklearn``
   directly.
2. ``features/`` is pure: no network, no fitted state. It turns a URL and bytes into
   raw values, emitting ``NaN`` for anything it cannot determine.
3. ``fetch/`` is the only module permitted to open a socket.
4. ``preprocess/`` owns all fitted state. ``features/`` produces ``NaN``s;
   ``preprocess/`` decides what they become.
"""

__version__ = "1.0.0"
ARTIFACT_SCHEMA_VERSION = "v1"

__all__ = ["ARTIFACT_SCHEMA_VERSION", "__version__"]
