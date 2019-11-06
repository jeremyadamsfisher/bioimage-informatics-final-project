# grim reaper data processing pipeline

PY=pipenv run python

RAW_IMAGES_DIR=/Users/jeremyfisher/Downloads/TCGA_DATA/
CONVERTED_PNG_IMAGES_DIR=./outdir/converted_images/
SPLIT_DATA_DIR=./outdir/split_data/
LATENT_ENCODINGS_FP=./outdir/image_encodings.csv
IMAGE_METADATA=./data/histology_image_annotations.csv

default: convert split

convert:
	$(PY) ./scripts/preprocessing/svs2png.py \
		-s $(RAW_IMAGES_DIR) \
		-o $(CONVERTED_PNG_IMAGES_DIR)

split:
	$(PY) ./scripts/preprocessing/split_data.py \
		'0.5,0.3,0.2' \
		-i $(CONVERTED_PNG_IMAGES_DIR) \
		-o $(SPLIT_DATA_DIR) \
		
autoencoder:
	$(PY) ./scripts/analysis/autoencoder.py \
		--train-dir $(SPLIT_DATA_DIR)/train \
		--test-dir $(SPLIT_DATA_DIR)/test \
		--valid-dir $(SPLIT_DATA_DIR)/valid \
		-o $(LATENT_ENCODINGS_FP)

survival:
	$(PY) ./scripts/analysis/survival.py \
		--encodings $(LATENT_ENCODINGS_FP) \
		--metadata $(IMAGE_METADATA)


push:
	git push --recurse-submodules=on-demand