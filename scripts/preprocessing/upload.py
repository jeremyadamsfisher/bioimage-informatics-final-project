import argparse
from google.cloud import storage
from pathlib import Path
from zipfile import ZipFile

parser = argparse.ArgumentParser()
parser.add_argument("--bucket-name", required=True)
parser.add_argument("--img-dir", type=Path, required=True)
opt = parser.parse_args()

storage_client = storage.Client()
bucket = storage_client.get_bucket(opt.bucket_name)

print("Compressing...")
img_compressed_f = Path("imgs.zip")
with ZipFile(img_compressed_f, "w") as zip:
    imgs = list(opt.img_dir.glob("*.png"))
    for i, fp in enumerate(imgs):
        print(f"...progress=({i}/{len(imgs)})")
        zip.write(str(fp))
print("Done compressing.")

print("Uploading...")
bucket.blob(img_compressed_f.name).upload_from_filename(str(img_compressed_f))
print("Done uploading.")

# clean up
img_compressed_f.unlink()