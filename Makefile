MAKEFLAGS += --no-print-directory
RUN_SCRIPTS := $(sort $(wildcard scripts/run_*.sh))
RUN_NAMES   := $(patsubst scripts/run_%.sh,%,$(RUN_SCRIPTS))

.PHONY: help \
		setup \
		run \
		run-help \
		format \
		check-format \
		gurobi-key \
		pre-commit-install \
		lock

# =================================================================================
# General
# =================================================================================
help: # Prints this message
	@awk -F': #' '/^[a-zA-Z0-9_-]+: #/ {printf "%-20s %s\n", $$1, $$2}' Makefile


# =================================================================================
# Setup
# =================================================================================
setup: # Install dependencies and prepare the repo (uv sync)
	@command -v uv >/dev/null 2>&1 || { echo "uv not found. Install it first (https://github.com/astral-sh/uv)."; exit 1; }
	@uv run pre-commit install
	@echo "Pre-commit hooks installed successfully"
	uv sync

# =================================================================================
# Checks and formatting
# =================================================================================
format: # Formats code
	@echo "Formatting Python code with uv..."
	uv format --preview-features format
	@echo "Code formatted."

check-format: # Checks code format
	@uv run ruff format --check .

check-typing: # Checks code types
	@uv run ty check

check-lock: # Checks uv.lock file
	@uv lock --check

check: # Executes all available checks
	@make check-format
	@make check-typing
	@make check-lock
	
# =================================================================================
# Gurobi
# =================================================================================
gurobi-key: # To set up Gurobi license key (requires GUROBI_KEY env var)
	@if [ -z "$(GUROBI_KEY)" ]; then \
		echo "GUROBI_KEY environment variable is not set. Please set it to your Gurobi license key => 'export GUROBI_KEY=<your_key>'"; \
		exit 1; \
	fi
	@grbgetkey $(GUROBI_KEY) 

# =================================================================================
# Run scenarios
# =================================================================================

RUN ?= help
run: # Run predefined scenarios; use RUN=<name> or run-<name> (see run-help)
	@$(MAKE) run-$(RUN)

$(addprefix run-,$(RUN_NAMES)):
	@bash scripts/run_$(patsubst run-%,%,$@).sh

run-help: # Show available RUN options for the run target
	@echo "Available RUN options:"
	@printf "  RUN=%s\n" "help"
	@for n in $(RUN_NAMES); do printf "  RUN=%s\n" "$$n"; done