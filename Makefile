# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
init:
	python3 -m pip install -r requirements.txt

prompts:
	python3 -m prompts

clean:
	rm -r prompts/__cache__

.PHONY: init prompts clean