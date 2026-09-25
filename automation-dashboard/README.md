# Anki dashboard module

This directory contains the Anki-specific dashboard service, page, and tests.
Mount `static/anki.html`, `static/anki.js`, `static/anki.css`, and `static/style.css` in the dashboard server and route `/api/anki` to `anki_service.status()` / `anki_service.update()`.

The module reads the crawler's local summary and config files, exposes enable/disable and run-now controls, and does not contain credentials.
