#!/bin/sh
echo "Run re-bayer on imagenet_val_5 dataset"
python re-bayer.py -i ./test_data/imagenet_val_5  -o ./test_data/imagenet_val_5_bayer_rggb/ -p RGGB
echo "Validate the re-bayer results"
python validate.py -i ./test_data/imagenet_val_5  -o ./test_data/imagenet_val_5_bayer_rggb -p RGGB -m 15 -r validation_results.csv -e ./view-errors.sh
echo "Check the images with PSNR lower than requested"
./view-errors.sh
