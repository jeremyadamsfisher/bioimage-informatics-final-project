# grim reaper data processing pipeline

PY=$(BIOIMAGE_PY_PATH)

BUCKET_NAME_GLCOUD=grim-reaper-initial-dataset
IMGS_PATH_GCP=gs://$(BUCKET_NAME_GLCOUD)/*.png
CREDENTIALS_GCLOUD=cred.json

MANIFEST_FP=gdc_manifest.2019-11-20_GW-FFPE-MCD.txt
INTERMEDIARY_DIR=intermediary
CONVERTED_PNG_IMAGES_DIR=$(INTERMEDIARY_DIR)/converted_images/
SPLIT_DATA_DIR=$(INTERMEDIARY_DIR)/split_data
LATENT_ENCODINGS_FP=./data/image_encodings.csv
IMAGE_METADATA=./data/histology_image_annotations.csv
SUPER_DATASET_FP=./data/dataset.csv
N_EPOCHS=3

IMG_SIZE_MIN=0
IMG_SIZE_MAX=5000

TRAIN_TEST_SPLIT="0.5,0.5"

default:
	echo "run either preprocess or pipeline!"

preprocess: download_and_convert_from_tcga

pipeline: download_from_bucket filter_unacceptable_imgs split autoencoder superdataset survival

download_and_convert_from_tcga:
	# download from TCGA and convert SVS images into PNGs for ingest
	# into PyTorch
	$(PY) ./scripts/preprocessing/download_from_tcga.py \
		--manifest-fp $(MANIFEST_FP) \
		--bucket-name $(BUCKET_NAME_GLCOUD) \
		--gcloud-credentials $(CREDENTIALS_GCLOUD)

download_from_bucket:
	mkdir -p $(CONVERTED_PNG_IMAGES_DIR) \
	&& gsutil -m cp "$(IMGS_PATH_GCP)" $(CONVERTED_PNG_IMAGES_DIR)

filter_unacceptable_imgs:
	$(PY) ./scripts/preprocessing/exclude.py \
		--min-x $(IMG_SIZE_MIN) --max-x $(IMG_SIZE_MAX) \
		--min-y $(IMG_SIZE_MIN) --max-y $(IMG_SIZE_MAX) \
		--img-dir $(CONVERTED_PNG_IMAGES_DIR) \
		--excluded-img-dir $(INTERMEDIARY_DIR)/excluded

split:
	# data needs to be split into training, validation and
	# testing; test set will be used for survival analysis
	$(PY) ./scripts/preprocessing/split_data.py \
		$(TRAIN_TEST_SPLIT) \
		-i $(CONVERTED_PNG_IMAGES_DIR) \
		-o $(SPLIT_DATA_DIR) \
		
autoencoder:
	# find latent representation space, extract latent encodings
	# from test dataset
	$(PY) ./scripts/analysis/train_predict_autoencoder.py \
		--train-dir $(SPLIT_DATA_DIR)/train \
		--test-dir $(SPLIT_DATA_DIR)/test \
		--epochs $(N_EPOCHS) \
		--img-size-max $(IMG_SIZE_MAX) \
		-o $(LATENT_ENCODINGS_FP)

superdataset:
	# build a dataset including latent representation and
	# clinically relevant metadata (days to death, primary diagnosis)
	$(PY) ./scripts/analysis/merge_img_data_and_latent_rep.py \
		--img-data-fp $(IMAGE_METADATA) \
		--latent-rep-fp $(LATENT_ENCODINGS_FP) \
		-o $(SUPER_DATASET_FP)

survival:
	# survival analysis
	$(PY) ./scripts/analysis/survival.py \
		--dataset $(SUPER_DATASET_FP)