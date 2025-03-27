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
	@echo "Activating venv..."
	poetry env activate
