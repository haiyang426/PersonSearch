# Norm-Aware Embedding for Efficient Person Search

## Results
|   Backbone   |  dataset | mAP | Top-1| config |
|--------------|---------|-----|------|--------|
|   R-50       | PRW | 45.4| 81.6 | [config](prw_nae.py)|
|   R-50       | CUHK-SYSU|91.6|92.6|[config](cuhk_nae.py)|

## train on PRW
    single GPU
    python tools/train.py configs/nae/prw_nae.py #PRW dataset

    multi GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/nae/prw_nae.py 4

## train on CUHK-SYSU
    single GPU
    python tools/train.py configs/nae/cuhk_nae.py #PRW dataset

    multi GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/nae/cuhk_nae.py 4
