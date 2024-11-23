# person search with mmdetection

## Installation
    conda create --name ps python=3.10 -y  
    conda activate ps  
    conda install pytorch torchvision -c pytorch

    pip install -U openmim  
    mim install mmengine  
    mim install "mmcv>=2.0.0"

    git clone https://github.com/haiyang426/person_search_with_mmdetection  
    cd person_search_with_mmdetection  
    pip install -v -e .

## dataset
    download PRW and CUHK-SYSU into data
    ---person search with mmdetection
      |---data
         |---PRW
         |---CUHK-SYSU

    mkdir annotations_coco
    python tools/dataset_converters/prw2coco.py -i data/PRW -o data/PRW/annotations_coco

    mkdir annotations_coco
    python tools/dataset_converters/cuhk2coco.py -i data/CUHK-SYSU/ -o data/CUHK-SYSU/annotations_coco

More installation details can be viewed [here](https://mmdetection.readthedocs.io/en/latest/get_started.html)

## model
[nae](configs/nae/README.md),  [SeqNet](configs/seqnet/README.md)

## To be completed 待完成
1. Support for the PoseTrack21 and MovieNet-PS dataset (PoseTrack21和MovieNet-PS数据集的支持)
2. SeqNet, COAT, PSTR, Align, CGPS, R-SiamNet等模型的复现
3. CBGM算法

## Thanks

[mmdetection](https://github.com/open-mmlab/mmdetection) 
[NAE4PS](https://github.com/dichen-cd/NAE4PS)