.PHONY: clean poetry install activate

clean:
	rm -rf ./.venv/ poetry.lock

poetry:
	@if ! which poetry; then \
		echo "Installing Poetry...";	 \
		curl -sSL https://install.python-poetry.org | python3 -; \
	else \
		echo "Poetry is already installed"; \
	fi

install: poetry
	@echo "Installing dependencies..."
	poetry install

activate:
	@echo "Entering environment shell"
	poetry shell

deactivate:
	@echo "Exiting environment shell"
	exit

help:
	@echo "Available commands:"
	@echo "  make install       - Install project dependencies"
	@echo "  make clean         - Remove virtual environment and lockfile"
	@echo "  make poetry        - Installs Poetry if not installed"
	@echo "  make activate      - Enter shell of Poetry virtual environment"
	@echo "  make deactivate    - Exit shell of Poetry virtual environment"
