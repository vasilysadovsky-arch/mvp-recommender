.PHONY: all eval data
data:
\tpython data_synth/generate.py
eval:
\tpython -m eval.run_eval
all: data eval
