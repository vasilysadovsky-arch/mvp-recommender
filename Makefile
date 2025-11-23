.PHONY: all eval data
data:
	python data_synth/generate.py
eval:
	python -m eval.run_eval
all: data eval
