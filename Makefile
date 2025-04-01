.PHONY: clean poetry install activate add-dependency deactivate help

clean: ## Remove virtual environment and lockfile
	rm -rf ./.venv/ poetry.lock

poetry: ## Installs Poetry if not already installed
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "Installing Poetry...";	 \
		curl -sSL https://install.python-poetry.org | python3 -; \
	else \
		echo "Poetry is already installed"; \
	fi

install: poetry ## Installs project dependencies
	@echo "Installing dependencies..."
	poetry install

activate: ## Enter shell of activated Poetry venv
	@echo "Entering environment shell"
	poetry shell

deactivate: ## Exit shell of activated Poetry venv
	@echo "Exiting environment shell"
	@exit

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

