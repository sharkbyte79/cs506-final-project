.PHONY: poetry install activate

poetry:
	@echo installing poetry
	curl -sSL https://install.python-poetry.org | python3 -

install: poetry
	@echo Installing dependencies
	poetry install

activate:
	@echo "Activating venv"
	poetry env activate
