"""Entry point for the PyInstaller build (the package's own __main__ uses relative imports)."""
from downloader.__main__ import main

raise SystemExit(main())
