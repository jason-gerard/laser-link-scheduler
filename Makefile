MAKEFLAGS += --no-print-directory

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
	uv sync
	
pre-commit-install: # Installs pre-commit and configure hooks
	@uv run pre-commit install
	@echo "Pre-commit hooks installed successfully"


# =================================================================================
# Checks and formatting
# =================================================================================
format: # Formats code
	@echo "Formatting Python code with uv..."
	uv format --preview-features format
	@echo "Code formatted."

check-format: # Checks code formatting
	@echo "Checking Python code formatting with uv..."
	uv format --check --preview-features format
	@echo "All files are formatted correctly."

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
run ?= help

run: # Run predefined scenarios; use RUN=<name> (see run-help)
	@case "$(RUN)" in \
		help) $(MAKE) run-help ;; \
		all) bash scripts/run_all.sh ;; \
		gs) bash scripts/run_all_gs.sh ;; \
		gs-inc) bash scripts/run_all_gs_inc.sh ;; \
		gs-inc-fast) bash scripts/run_all_gs_inc_fast.sh ;; \
		lls) bash scripts/run_all_lls.sh ;; \
		*) echo "Unknown RUN=$(RUN)"; $(MAKE) run-help; exit 1 ;; \
	esac

run-help: # Show available RUN options for the run target
	@echo "Available RUN options:"
	@printf "  RUN=%-11s %s\n" "help" "Show this message"
	@printf "  RUN=%-11s %s\n" "all" "Run all base Mars/Earth scenarios (schedulers => fcp/random/alternating/lls)"
	@printf "  RUN=%-11s %s\n" "gs" "Run all GS scenarios (schedulers => fcp/random/alternating/lls/lls_pat_unaware/lls_mip/lls_lp)"
	@printf "  RUN=%-11s %s\n" "gs-inc" "Run all GS incremental reduced scenarios (schedulers => fcp/random/alternating/lls/lls_pat_unaware/lls_mip)"
	@printf "  RUN=%-11s %s\n" "gs-inc-fast" "Run a smaller GS incremental set for quick experiments"
	@printf "  RUN=%-11s %s\n" "lls" "Run all base Mars/Earth scenarios with LLS only"
