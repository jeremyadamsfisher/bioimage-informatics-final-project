FROM pytorch/pytorch:latest
RUN apt-get update && \
    apt-get install -y openslide-tools
RUN pip install openslide-python
RUN pip install numpy pandas
RUN mkdir -p /outdir/converted_images \
        /outdir/image_encodings \
        /outdir/split_data
ADD models ./models
ADD scripts ./scripts
ADD Makefile ./