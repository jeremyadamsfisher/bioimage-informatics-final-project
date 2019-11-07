"""merge the latent encodings derived from autoencoder
with the survival time and diagnosis extracted from scraping 
the TCGA api"""

import argparse
import pandas as pd
from pathlib import Path

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-data-fp", type=Path, required=True)
    parser.add_argument("--latent-rep-fp", type=Path, required=True)
    parser.add_argument("-o", "--out-df-fp", dest="out_df_fp", type=Path, required=True)
    return parser.parse_args().__dict__


def main(img_data_fp, latent_rep_fp, out_df_fp):
    df_annot = pd.read_csv(img_data_fp)
    df_latent = (
        pd.read_csv(latent_rep_fp)
        .assign(prefix=lambda _df: _df.img_fp.apply(lambda s: s[:12]))
    )
    df = df_annot.merge(df_latent, on="prefix")
    df.to_csv(out_df_fp)
    

if __name__ == "__main__":
    main(**cli())