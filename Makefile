VENV = .rlvenv
IN_DOCKER = 0

ifdef IN_DOCKER
PYTHON = python
PIP = pip
TORCH = torch --index-url https://download.pytorch.org/whl/cpu
else
PYTHON = .rlvenv/local/bin/python
PIP = .rlvenv/local/bin/pip
TORCH = torch
endif

DOCKER_IMG_NAME = instadeep-rl

install:
	apt-get install -y swig
	$(PIP) install --upgrade pip
	$(PIP) install $(TORCH)
	$(PIP) install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +

lint:
	$(PYTHON) -m pylint --disable=R,C *.py

format:
	$(PYTHON) -m black .

train:
	$(PYTHON) train.py $(AGENT)

evaluate:
	$(PYTHON) eval.py $(AGENT)

analysis:
	$(PYTHON) analysis.py

build:
	docker build -t $(DOCKER_IMG_NAME) .

run:
	$(eval IN_DOCKER=1)
	docker run -it -v $(PWD)/checkpoints:/app/checkpoints --rm -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix $(DOCKER_IMG_NAME) /bin/bash
