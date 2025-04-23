# Python-CVIT

(Implemented and run on Ubuntu (WSL))

```bash
cd conv_vision_transformer/
```

```bash
python train.py -d training_data/ -e 10 -b 32

# giving data directory is necessary, epoch by default 1 and batch size 32

python predict.py --model-path <path>

# giving model path is necessary

# for example :

python predict.py --model-path saved_model/trained_model.pth

```

Model paths :

one path : `saved_model/trained_model.pth`

other path: `weight/deepfake_cvit_gpu_inference_ep_50.pth`
