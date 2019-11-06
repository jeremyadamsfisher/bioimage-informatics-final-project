# grim reaper data processing pipeline

PY=pipenv run python

DATADIR=$(PWD)
RAW_IMAGES_DIR=./outdir/raw_iamges
CONVERTED_PNG_IMAGES_DIR=./outdir/converted_images
LATENT_ENCODINGS_FP=./outdir/image_encodings
IMAGE_METADATA=./data/histology_image_annotations.csv

default: convert autoencoder survival

convert:
	$(PY) ./scripts/preprocessing/svs2png.py \
		-s $(RAW_IMAGES_DIR) \
		-o $(CONVERTED_PNG_IMAGES_DIR)

autoencoder:
	$(PY) ./scripts/analysis/autoencoder.py \
		'0.5,0.3,0.2' \
		-i $(CONVERTED_PNG_IMAGES_DIR) \
		-o $(LATENT_ENCODINGS_FP)

survival:
	$(PY) ./scripts/analysis/survival.py \
		--encodings $(LATENT_ENCODINGS_FP) \
		--metadata $(IMAGE_METADATA)
