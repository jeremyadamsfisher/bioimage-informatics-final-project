# grim reaper data processing pipeline

PY=pipenv run python

DATADIR=$(PWD)
OUTDIR=./outdir

default: convert autoencoder survival

convert:
	mkdir -p ./output
	$(PY) ./scripts/preprocessing/svs2png.py \
		-s $(DATADIR) \
		-o $(OUTDIR)

autoencoder:
	$(PY) ./scripts/analysis/autoencoder.py \
		'0.5,0.3,0.2' \
		
		-o $(OUTDIR) \


survival:
	$(PY) ./scripts/analysis/survival.py