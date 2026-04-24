DRAWIO     := /snap/bin/drawio
DIAGRAMS   := docs/diagrams
DRAWIO_SRC := $(wildcard $(DIAGRAMS)/*.drawio)
DRAWIO_PNG := $(DRAWIO_SRC:.drawio=.drawio.png)

.PHONY: test e2e clean dev diagrams

dev:
	poetry install

test:
	poetry run pytest tests/ -v

e2e:
	poetry run bash scripts/e2e_test.sh

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

diagrams: $(DRAWIO_PNG)

$(DIAGRAMS)/%.drawio.png: $(DIAGRAMS)/%.drawio
	xvfb-run $(DRAWIO) -x -f png -e -b 10 -o $@ $<
