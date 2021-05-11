# MultiFix transformer test

## Usage
### Data Processing
```
    python preprocess.py -data_name DeepFix -save_data multifix.pkl -share_vocab
```

### Training
```
    python train.py -data_pkl multifix.pkl -embs_share_weight -proj_share_weight -label_smoothing -output_dir output
```

### Prediction
```
    python multifix.py -data_pkl multifix.pkl -model output/model.chkpt -output prediction.txt
```
