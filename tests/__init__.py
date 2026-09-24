"""Headless tests. Run each from the project root:  python -m tests.<name>

Every client a test connects looks for the host's map in tests/ and keeps any
map it has to download in a throwaway folder, so running the tests never
writes into maps/."""
import atexit
import shutil
import tempfile

import net.client as _client

_client.MAPS_DIR = "tests"
_client.DOWNLOAD_DIR = tempfile.mkdtemp(prefix="bangbang_downloads_")
atexit.register(shutil.rmtree, _client.DOWNLOAD_DIR, True)
