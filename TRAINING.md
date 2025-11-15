# Training Guide

## Prerequisites

- Đã hoàn thành setup trong SETUP.md
- Docker image `ielts-trainer` đã được build
- GPU available trong Docker

## 1. Chuẩn bị Dataset

### Từ audio files

Đặt file audio vào `data training/` folder, sau đó:

```bash
python app/process_audio.py
```

Hoặc chạy API để upload audio và tự động xử lý.

### Tạo training dataset

```bash
python app/prepare_dataset.py
```

Dataset được tạo tại: `data/ielts_training_data.jsonl`

Kiểm tra:

```bash
python app/prepare_dataset.py --validate
```

## 2. Chạy Training

### Quick Start

```bash
docker-compose --profile training up trainer
```

### Custom Parameters

```bash
docker run --rm --gpus all \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/models:/app/models \
  -v ${PWD}/config:/app/config \
  ielts-trainer \
  python3 app/train_qlora.py --model qwen --epochs 5 --batch-size 1
```

### Options:

- `--model`: qwen | phi2 | gemma
- `--epochs`: số epoch training (default: 3)
- `--batch-size`: batch size (default: 1 cho 4GB GPU)
- `--output`: đường dẫn lưu model (default: ./models/ielts-finetuned)
- `--dataset`: đường dẫn dataset (default: ./data/ielts_training_data.jsonl)

## 3. Monitor Progress

### Xem logs

```bash
docker logs -f ielts-trainer
```

### Xem GPU usage

```bash
nvidia-smi -l 1
```

### Container stats

```bash
docker stats ielts-trainer
```

## 4. Training Timeline

**Qwen 1.5B trên RTX 3050 4GB:**

- Download model: ~10 phút (lần đầu)
- Load & setup: ~2 phút
- Epoch 1: ~15-20 phút
- Epoch 2: ~15-20 phút
- Epoch 3: ~15-20 phút
- **Tổng: ~50-70 phút**

## 5. Kết quả

Model được lưu tại: `models/ielts-finetuned/`

Files:

```
models/ielts-finetuned/
├── adapter_config.json
├── adapter_model.safetensors
├── config.json
├── tokenizer.json
└── tokenizer_config.json
```

## 6. Test Model

```bash
docker run --rm --gpus all \
  -v ${PWD}/models:/app/models \
  ielts-trainer \
  python3 app/inference.py \
    --backend autotrain \
    --model /app/models/ielts-finetuned \
    --transcript "Technology has changed our lives significantly."
```

## 7. Deploy Inference

### Start inference server

```bash
docker-compose --profile inference up -d inference
```

### Test API

```bash
curl -X POST http://localhost:8001/score \
  -H "Content-Type: application/json" \
  -d '{"transcript": "I think education is very important."}'
```

## 8. Convert Model (Optional)

### Convert to GGUF (CPU inference)

```bash
python app/convert_model.py \
  --model ./models/ielts-finetuned \
  --format gguf \
  --quantization Q4_K_M
```

### Convert to AWQ (fast GPU inference)

```bash
python app/convert_model.py \
  --model ./models/ielts-finetuned \
  --format awq \
  --bits 4
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size, increase gradient accumulation
docker run ... ielts-trainer \
  python3 app/train_qlora.py --batch-size 1 --gradient-accumulation-steps 8
```

### CUDA not available

```bash
# Check GPU in container
docker run --rm --gpus all ielts-trainer \
  python3 -c "import torch; print(torch.cuda.is_available())"
```

### Training bị dừng giữa chừng

```bash
# Resume từ checkpoint
docker-compose --profile training up trainer
```

Checkpoints được lưu tại: `models/ielts-finetuned/checkpoint-*`

### Muốn train lại từ đầu

```bash
# Xóa model cũ
rm -rf models/ielts-finetuned

# Train lại
docker-compose --profile training up trainer
```

## Performance Notes

**RTX 3050 4GB:**

- Training speed: ~120 tokens/sec
- Memory usage: 3.8-3.9GB VRAM
- Batch size: 1 (optimal)
- Gradient accumulation: 4 (optimal)

**RTX 3060 8GB:**

- Batch size có thể tăng lên 2-4
- Training nhanh hơn ~50%
