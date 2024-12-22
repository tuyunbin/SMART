# SMART: Syntax-calibrated Multi-Aspect Relation Transformer for Change Captioning

This package contains the accompanying code for the following paper:

Tu, Yunbin, et al. ["SMART: Syntax-calibrated Multi-Aspect Relation Transformer for Change Captioning."](https://ieeexplore.ieee.org/document/10433795), which has appeared as a regular paper in IEEE TPAMI 2024.

## Installation
1. Clone this repository
2. cd SMART
1. Make virtual environment with Python 3.5 
2. Install requirements (`pip install -r requirements.txt`)
3. Setup COCO caption eval tools ([github](https://github.com/mtanti/coco-caption)) 
4. Two TITAN Xp GPUs or others.

## Data
1. Download data from here: [baidu drive link](https://pan.baidu.com/s/1PiLlUw5PP7IzWP_OH400AQ), where the extraction code is qtzy.

Extracting this file will create `data` directory and fill it up with CLEVR-Change dataset.

2. Preprocess data

* Extract visual features using ImageNet pretrained ResNet-101:
```
# processing default images
python scripts/extract_features.py --input_image_dir ./data/images --output_dir ./data/features --batch_size 128

# processing semantically changes images
python scripts/extract_features.py --input_image_dir ./data/sc_images --output_dir ./data/sc_features --batch_size 128

# processing distractor images
python scripts/extract_features.py --input_image_dir ./data/nsc_images --output_dir ./data/nsc_features --batch_size 128
```

* Build vocab and label files of SMARL (LSTM decoder) by using caption annotations:
```
python scripts/preprocess_captions_pos.py --input_captions_json ./data/change_captions.json --input_neg_captions_json ./data/no_change_captions.json --input_image_dir ./data/images --input_pos ./data/pos_token.pkl --split_json ./data/splits.json --output_vocab_json ./data/vocab.json --output_h5 ./data/labels.h5
```

* Build vocab and label files of SMART (transformer decoder) by using caption annotations:
```
python scripts/preprocess_captions_transformer_pos.py --input_captions_json ./data/change_captions.json --input_neg_captions_json ./data/no_change_captions.json --input_image_dir ./data/images --input_pos ./data/pos_token.pkl --split_json ./data/splits.json --output_vocab_json ./data/transformer_vocab.json --output_h5 ./data/transformer_labels.h5
```
You can skip the above process about building vocab and label, and download the preprocessed data here: [baidu drive link](https://pan.baidu.com/s/1qiZ8NKItzz91b2mamvjcfQ?pwd=ia5n), where the extraction code is ia5n.


## Training
To train the proposed method, run the following commands:
```
# create a directory or a symlink to save the experiments logs/snapshots etc.
mkdir experiments
# OR
ln -s $PATH_TO_DIR$ experiments

# this will start the visdom server for logging
# start the server on a tmux session since the server needs to be up during training
python -m visdom.server

# start training for SMARL
python train_lstm.py --cfg configs/dynamic/lstm.yaml  --entropy_weight 0.0001


# start training for SMART
python train_trans.py --cfg configs/dynamic/transformer.yaml  
```

## Testing/Inference for SMARL
To test/run inference on the test dataset, run the following command
```
python test_lstm.py --cfg configs/dynamic/lstm.yaml  --snapshot 9000 --gpu 1
```

## Testing/Inference for SMART
To test/run inference on the test dataset, run the following command
```
python test_trans.py --cfg configs/dynamic/transformer.yaml  --snapshot 6000 --gpu 1
```


## Evaluation
* Caption evaluation

```
# This will run evaluation on the results generated from the validation set and print the best results
python evaluate.py --results_dir ./experiments/SMARL/eval_sents --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
python evaluate.py --results_dir ./experiments/SMART/eval_sents --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```

Once the best model is found on the validation set, you can run inference on test set for that specific model using the command exlpained in the `Testing/Inference` section and then finally evaluate on test set:
```
python evaluate.py --results_dir ./experiments/SMARL/test_output/captions --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
python evaluate.py --results_dir ./experiments/SMART/test_output/captions --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```
The results are saved in `./experiments/SMARL(SMART)/test_output/captions/eval_results.txt`


If you find this helps your research, please consider citing:
```
@article{tu2024smart,
  title={SMART: Syntax-Calibrated Multi-Aspect Relation Transformer for Change Captioning},
  author={Tu, Yunbin and Li, Liang and Su, Li and Zha, Zheng-Jun and Huang, Qingming},
  journal={IEEE Transactions on Pattern Analysis \& Machine Intelligence},
  volume={46},
  number={07},
  pages={4926--4943},
  year={2024},
  publisher={IEEE Computer Society}
}
```

## Contact
My email is tuyunbin1995@foxmail.com.



