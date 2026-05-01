DRAWIO     := /snap/bin/drawio
DIAGRAMS   := docs/diagrams
DRAWIO_SRC := $(wildcard $(DIAGRAMS)/*.drawio)
DRAWIO_PNG := $(DRAWIO_SRC:.drawio=.drawio.png)

.PHONY: test typecheck e2e e2e-cli e2e-container e2e-claude clean dev diagrams

dev:
	poetry install

test:
	poetry run pytest tests/ -v --ignore=tests/e2e

typecheck:
	poetry run mypy --strict --ignore-missing-imports \
	  --follow-imports=silent \
	  src/memman/store/backend.py \
	  src/memman/store/factory.py \
	  src/memman/store/model.py \
	  src/memman/store/sqlite.py \
	  src/memman/graph/bfs.py \
	  src/memman/graph/semantic.py \
	  src/memman/graph/engine.py

e2e:
	poetry run pytest tests/e2e/ -v

e2e-cli:
	poetry run pytest tests/e2e/ -v -m e2e_cli

e2e-container:
	poetry run pytest tests/e2e/ -v -m "e2e_container and not live_claude and not systemd"

e2e-claude:
	poetry run pytest tests/e2e/ -v -m live_claude

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

diagrams: $(DRAWIO_PNG)

$(DIAGRAMS)/%.drawio.png: $(DIAGRAMS)/%.drawio
	xvfb-run $(DRAWIO) -x -f png -e -b 10 -o $@ $<
