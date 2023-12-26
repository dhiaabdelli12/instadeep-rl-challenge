install:
	pip install --upgrade pip
	pip install -r requirements.txt

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -r .pytest_cache

lint:
	pylint --disable=R,C *.py

format:
	black .

train:
	python train.py

evaluate:
	python eval.py
