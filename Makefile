run:
	uvicorn app.main:app --port 8000 --reload
test:
	pytest -q
demo:
	python -m scripts.offline_demo
