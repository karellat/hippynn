SHELL := /bin/bash
.ONESHELL:
.SILENT:

.DEFAULT_GOAL := train

ENV_NAME := hippy-omol
CONFIG ?= config.yaml
ROOT_DIR := ../..
TRAIN_SCRIPT := 02_hiphop_omol_dpp.py
PROFILE_SCRIPT := 03_hiphop_omol_profiler.py
STAMP_DIR := .make
EDITABLE_STAMP := $(STAMP_DIR)/hippynn-editable.stamp
EDITABLE_SOURCES := $(shell find $(ROOT_DIR)/hippynn -type f) \
	$(ROOT_DIR)/pyproject.toml \
	$(ROOT_DIR)/setup.py \
	$(ROOT_DIR)/setup.cfg

define ACTIVATE_ENV
set -euo pipefail
if command -v conda >/dev/null 2>&1; then
	eval "$$(conda shell.bash hook)"
	if ! conda env list | awk '{print $$1}' | grep -Fxq "$(ENV_NAME)"; then
		echo "Conda environment '$(ENV_NAME)' does not exist." >&2
		exit 1
	fi
	conda activate "$(ENV_NAME)"
else
	echo "Could not find conda to activate '$(ENV_NAME)'." >&2
	exit 1
fi
endef

.PHONY: help env-check train profile reinstall

help:
	printf '%s\n' \
		"make train [CONFIG=path/to/config.yaml]" \
		"make profile [CONFIG=path/to/config.yaml]" \
		"make reinstall"

env-check:
	$(ACTIVATE_ENV)
	python --version

$(EDITABLE_STAMP): $(EDITABLE_SOURCES)
	mkdir -p "$(STAMP_DIR)"
	$(ACTIVATE_ENV)
	pip install -e "$(ROOT_DIR)"
	touch "$(abspath $@)"

reinstall:
	rm -f "$(EDITABLE_STAMP)"
	$(MAKE) "$(EDITABLE_STAMP)"

train: $(EDITABLE_STAMP)
	$(ACTIVATE_ENV)
	test -f "$(CONFIG)" || { echo "Config file not found: $(CONFIG)" >&2; exit 1; }
	python "$(TRAIN_SCRIPT)" --config "$(CONFIG)"

profile: $(EDITABLE_STAMP)
	$(ACTIVATE_ENV)
	test -f "$(CONFIG)" || { echo "Config file not found: $(CONFIG)" >&2; exit 1; }
	python "$(PROFILE_SCRIPT)" --config "$(CONFIG)"
