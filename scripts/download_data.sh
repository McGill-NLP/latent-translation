#!/bin/bash

set -e

SCRIPTS_DIR=$PWD
MAIN_DIR=$(dirname `pwd`)
DATA_DIR=$MAIN_DIR/data

# download XNLI dataset
OUTPATH=$DATA_DIR/xnli-tmp/
if [ ! -d $OUTPATH/XNLI-MT-1.0 ]; then
  if [ ! -f $OUTPATH/XNLI-MT-1.0.zip ]; then
    wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip -P $OUTPATH -q --show-progress
  fi
  unzip -qq $OUTPATH/XNLI-MT-1.0.zip -d $OUTPATH
fi
if [ ! -d $OUTPATH/XNLI-1.0 ]; then
  if [ ! -f $OUTPATH/XNLI-1.0.zip ]; then
    wget -c https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip -P $OUTPATH -q --show-progress
  fi
  unzip -qq $OUTPATH/XNLI-1.0.zip -d $OUTPATH
fi
python $SCRIPTS_DIR/utils_preprocess.py \
  --data_dir $OUTPATH \
  --output_dir $DATA_DIR/xnli/ \
  --task xnli
rm -rf $OUTPATH
echo "Successfully downloaded data at $DATA_DIR/xnli"

# download PAWS-X dataset
cd $DATA_DIR
wget https://storage.googleapis.com/paws/pawsx/x-final.tar.gz -q --show-progress
tar xzf x-final.tar.gz -C $DATA_DIR/
python $SCRIPTS_DIR/utils_preprocess.py \
  --data_dir $DATA_DIR/x-final \
  --output_dir $DATA_DIR/pawsx/ \
  --task pawsx
rm -rf x-final x-final.tar.gz
echo "Successfully downloaded data at $DATA_DIR/pawsx"

# download XCOPA dataset
cd $DATA_DIR
git clone https://github.com/cambridgeltl/xcopa.git xcopa-tmp
mkdir -p $DATA_DIR/xcopa
mv xcopa-tmp/data/*/*.jsonl xcopa
rm -rf xcopa-tmp
echo "Successfully downloaded data at $DATA_DIR/xcopa"
