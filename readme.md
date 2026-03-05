# 🚀 BitSC: Bit-Level Semantic Communication Framework

This folder contains the pretrained BitSC models under **error-free channel conditions** for different code compression ratios (cpp). These models can later be **fine-tuned on noisy channels**.

## 📊 Model Configurations

| C   | k        | L_b          |
|-----|----------|--------------|
| 16  | 1/32     | 16 × 256 = 4096  |
| 32  | 1/16     | 32 × 256 = 8192  |
| 64  | 1/8      | 64 × 256 = 16384 |

## 🏋️ Training Example (C=32, k=1/16)

```bash
python main.py \
  --training \
  --gpu 0 \
  --C 32 \
  --lr 1e-4 \
  --step 200 \
  --min_lr 5e-6 \
  --pass_channel \
  --px BitSC \
  --snr_min -3 \
  --snr_max 6 \
  --pretrain "Pretrain/WITT_BitSC_C16.model"
```
> 💡 **Note:** This is a helpful note for the user.--pretrain specifies the path to the pretrained model you want to start from (here C=16).

## 🧪 Testing Example (C=32, k=1/16)
```bash
python main.py \
  --training \
  --gpu 0 \
  --C 32 \
  --pass_channel \
  --pretrain <path_to_your_trained_model> \
  --test_snr 6
```

## ⚙️ Environment Setup
Install required Python packages:
```bash
pip install -r requirements.txt
```