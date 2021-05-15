# MultiFix transformer test

## Usage

### Data Processing
Download the dataset that has already been preprocessed
```
    bash download_processing_data.sh
```

### Training
```
    python train.py -d $DATA_NAME -es -ps -ls -o output
```

If using Synchronized positional embedding:
```
    python train.py -d $DATA_NAME -es -ps -ls -o output -sp -add
```
If using with synchronized positional embeddings concatenated
```
    python train.py -d $DATA_NAME -es -ls -o output -sp -dw 100
```

Or if using with Synchronized positional embedding added and positional encoding
```
    python train.py -d $DATA_NAME -es -ps -ls -o output -sp -wsp -add
```
Or if using with Synchronized positional embedding concatenated and positional encoding
```
    python train.py -d $DATA_NAME -es -ls -o output -sp -wsp -dw 100
```

### Prediction
```
    python run_multifix.py -d $DATA_NAME (opt: -sp, -wsp, -add ...)
```

### Code Repair Test
```
    python test_multifix.py
```
