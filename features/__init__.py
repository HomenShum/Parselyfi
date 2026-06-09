"""ParselyFi feature tabs package.

Shared support lives in ``features.common``. The individual feature tabs
(company_research, news_youtube, transcription) import from there.

This package intentionally exposes nothing at import time so that importing
``features`` never pulls in optional/heavy dependencies.
"""
