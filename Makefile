# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
SHELL:=/bin/bash

init: deinit
	python3 -m venv .env
	source .env/bin/activate && \
	python3 -m pip install -r requirements.txt
	touch .env/COLCON_IGNORE

deinit:
	rm -rf .env

prompts:
	python3 -m prompts

clean:
	rm -r prompts/__cache__

.PHONY: init deinit prompts clean
