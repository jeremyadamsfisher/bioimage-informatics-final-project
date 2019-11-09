# grim reaper data processing pipeline

PY=$(BIOIMAGE_PY_PATH)

DOCKER_IMG_NAME=bioimage_informatics_final_project_runtime

BUCKET_NAME_GLCOUD=grim-reaper-initial-dataset
IMGS_PATH_GCP=gs://grim-reaper-initial-dataset/imgs.zip

RAW_IMAGES_DIR=/Users/jeremyfisher/Downloads/TCGA_DATA/
CONVERTED_PNG_IMAGES_DIR=./outdir/converted_images/
SPLIT_DATA_DIR=./outdir/split_data
LATENT_ENCODINGS_FP=./outdir/image_encodings.csv
IMAGE_METADATA=./data/histology_image_annotations.csv
SUPER_DATASET_FP=./outdir/dataset.csv
N_EPOCHS=100

default:
	echo "run either preprocess or pipeline!"

preprocess: convert upload_to_bucket

pipeline: download_from_bucket split autoencoder superdataset

convert:
	# convert SVS images into PNGs for ingest into PyTorch
	# also, filter out images deriving from low quality biopsys
	ls $(RAW_IMAGES_DIR) && \
	$(PY) ./scripts/preprocessing/svs2png.py \
		-s $(RAW_IMAGES_DIR) \
		-o $(CONVERTED_PNG_IMAGES_DIR)

upload_to_bucket:
	$(PY) ./scripts/preprocessing/upload.py \
		--bucket-name $(BUCKET_NAME_GLCOUD) \
		--img-dir $(CONVERTED_PNG_IMAGES_DIR)


download_from_bucket:
	rm -rf $(CONVERTED_PNG_IMAGES_DIR) \
	&& gsutil cp $(IMGS_PATH_GCP) . \
	&& unzip *.zip \
	&& rm *.zip

split:
	# data needs to be split into training, validation and
	# testing; test set will be used for survival analysis
	ls $(CONVERTED_PNG_IMAGES_DIR) && \
	$(PY) ./scripts/preprocessing/split_data.py \
		'0.33,0.33,0.33' \
		-i $(CONVERTED_PNG_IMAGES_DIR) \
		-o $(SPLIT_DATA_DIR) \
		
autoencoder:
	# find latent representation space, extract latent encodings
	# from test dataset
	ls $(SPLIT_DATA_DIR)/* && \
	$(PY) ./scripts/analysis/autoencoder.py \
		--train-dir $(SPLIT_DATA_DIR)/train \
		--test-dir $(SPLIT_DATA_DIR)/test \
		--valid-dir $(SPLIT_DATA_DIR)/valid \
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

build:
	docker build -t $(DOCKER_IMG_NAME) .

run: build
	docker run \
		-v $(PWD)/data:/data \
		-v $(RAW_IMAGES_DIR):/raw \
		$(DOCKER_IMG_NAME) make pipeline \
			RAW_IMAGES_DIR=/raw \
			BIOIMAGE_PY_PATH=/opt/conda/bin/python