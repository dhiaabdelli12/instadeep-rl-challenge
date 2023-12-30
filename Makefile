VENV = .rlvenv
PYTHON = $(VENV)/local/bin/python
PIP = $(VENV)/local/bin/pip

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +

lint:
	$(PYTHON) -m pylint --disable=R,C *.py

format:
	$(PYTHON) -m black .

train:
	$(PYTHON) train.py

evaluate:
	$(PYTHON) eval.py

