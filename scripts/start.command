#!/bin/zsh
cd "${0:A:h:h}"
.venv/bin/python scripts/service.py start
open http://127.0.0.1:8787
