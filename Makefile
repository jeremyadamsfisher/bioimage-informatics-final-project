# grim reaper data processing pipeline

PY=$(BIOIMAGE_PY_PATH)

DOCKER_IMG_NAME=bioimage_informatics_final_project_runtime

BUCKET_NAME_GLCOUD=grim-reaper-initial-dataset
IMGS_PATH_GCP=gs://grim-reaper-initial-dataset/*.png
CREDENTIALS_GCLOUD=cred.json

MANIFEST_FP=gdc_manifest.2019-11-20_GW-FFPE-MCD.txt
RAW_IMAGES_DIR=/Users/jeremyfisher/Downloads/TCGA_DATA/
CONVERTED_PNG_IMAGES_DIR=./outdir/converted_images/
SPLIT_DATA_DIR=./outdir/split_data
LATENT_ENCODINGS_FP=./outdir/image_encodings.csv
IMAGE_METADATA=./data/histology_image_annotations.csv
SUPER_DATASET_FP=./outdir/dataset.csv
N_EPOCHS=100

default:
	echo "run either preprocess or pipeline!"

preprocess: download_and_convert_from_tcga

pipeline: download_from_bucket split autoencoder superdataset survival

download_and_convert_from_tcga:
	# download from TCGA and convert SVS images into PNGs for ingest
	# into PyTorch
	$(PY) ./scripts/preprocessing/download_from_tcga.py \
		--manifest-fp $(MANIFEST_FP) \
		--bucket-name $(BUCKET_NAME_GLCOUD) \
		--gcloud-credentials $(CREDENTIALS_GCLOUD)

clean:
	rm -rf $(CONVERTED_PNG_IMAGES_DIR)

download_from_bucket: clean
	gsutil cp $(IMGS_PATH_GCP) $(CONVERTED_PNG_IMAGES_DIR)/ \

split:
	# data needs to be split into training, validation and
	# testing; test set will be used for survival analysis
	$(PY) ./scripts/preprocessing/split_data.py \
		'0.5,0.5' \
		-i $(CONVERTED_PNG_IMAGES_DIR) \
		-o $(SPLIT_DATA_DIR) \
		
autoencoder:
	# find latent representation space, extract latent encodings
	# from test dataset
	$(PY) ./scripts/analysis/autoencoder.py \
		--train-dir $(SPLIT_DATA_DIR)/train \
		--test-dir $(SPLIT_DATA_DIR)/test \
		--epochs $(N_EPOCHS) \
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