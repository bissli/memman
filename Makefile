DRAWIO     := /snap/bin/drawio
DIAGRAMS   := docs/diagrams
DRAWIO_SRC := $(wildcard $(DIAGRAMS)/*.drawio)
DRAWIO_PNG := $(DRAWIO_SRC:.drawio=.drawio.png)

.PHONY: test test-postgres typecheck e2e e2e-cli e2e-container clean dev diagrams

dev:
	poetry install

test:
	poetry run pytest tests/ -v --ignore=tests/e2e

test-postgres:
	MEMMAN_BACKEND=postgres MEMMAN_PG_DSN=$$PG_TEST_URL \
	  poetry run pytest tests/ -v --ignore=tests/e2e -m postgres

typecheck:
	poetry run mypy --strict --ignore-missing-imports \
	  --follow-imports=silent \
	  src/memman/store/backend.py \
	  src/memman/store/factory.py \
	  src/memman/store/model.py \
	  src/memman/store/sqlite.py \
	  src/memman/store/postgres.py \
	  src/memman/store/db.py \
	  src/memman/store/node.py \
	  src/memman/store/edge.py \
	  src/memman/store/oplog.py \
	  src/memman/graph/bfs.py \
	  src/memman/graph/semantic.py \
	  src/memman/graph/engine.py \
	  src/memman/graph/temporal.py \
	  src/memman/graph/entity.py \
	  src/memman/graph/causal.py \
	  src/memman/embed/fingerprint.py \
	  src/memman/search/recall.py \
	  src/memman/pipeline/remember.py \
	  src/memman/doctor.py \
	  src/memman/maintenance.py

e2e:
	poetry run pytest tests/e2e/ -v

e2e-cli:
	poetry run pytest tests/e2e/ -v -m e2e_cli

e2e-container:
	poetry run pytest tests/e2e/ -v -m "e2e_container and not systemd"

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

diagrams: $(DRAWIO_PNG)

$(DIAGRAMS)/%.drawio.png: $(DIAGRAMS)/%.drawio
	xvfb-run $(DRAWIO) -x -f png -e -b 10 -o $@ $<
