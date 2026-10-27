# Security deployment changes

The webcam server now binds only to 127.0.0.1, disables the interactive debugger, and no longer permits cross-origin reads. Do not expose webcam routes through a reverse proxy without authentication.
