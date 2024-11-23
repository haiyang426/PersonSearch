# Sequential End-to-end Network for Efficient Person Search

## Results
|   Backbone   |  dataset |  mAP | Top-1 |       config         |
|--------------|--------- |------|-------|----------------------|
|   R-50       | PRW      | 48.0 |  83.2 | [config](prw_seqnet.py) |
|   R-50       | CUHK-SYSU| 93.5 |  94.2 | [config](cuhk_seqnet.py)|

## train on PRW
    single GPU
    python tools/train.py configs/seqnet/prw_seqnet.py #PRW dataset

    multi GPU
    CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/seqnet/prw_seqnet.py 2

## train on CUHK-SYSU
    single GPU
    python tools/train.py configs/nae/cuhk_nae.py #PRW dataset

    multi GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/nae/cuhk_nae.py 4
