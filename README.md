# MultiFix transformer test

## Usage

### Download the dataset
```
    bash download_dataset.sh
```

### Data Processing
```
    python preprocess.py -d DeepFix -sv
    python preprocess.py -d DrRepair_deepfix -sv
```

### Training
```
    python train.py -es -ps -ls -o output
```

If using Synchronized positional embedding:
```
    python train.py -es -ps -ls -o output -sp
```

Or if using Synchronized positional embedding with positional encoding:
```
    python train.py -es -ps -ls -o output -sp -wsp
```

### Prediction
```
    python run_multifix.py -data_pkl multifix.pkl -model output/model.chkpt -output prediction.txt
```

### Code Repair Test
TODO
