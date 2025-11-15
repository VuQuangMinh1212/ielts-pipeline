# Setup & Installation Guide

## Requirements

- Docker Desktop với WSL2 (Windows) hoặc Docker + NVIDIA Container Toolkit (Linux)
- NVIDIA GPU với 4GB+ VRAM (RTX 3050 hoặc tốt hơn)
- 20GB disk space

## 1. Clone Repository

```bash
git clone https://github.com/VuQuangMinh1212/ielts-pipeline.git
cd ielts-pipeline
```

## 2. Cấu hình Environment

```bash
cp .env.example .env
```

Chỉnh sửa `.env`:

```env
GENAI_API_KEY=your_gemini_api_key_here
MODEL_NAME=qwen
TRAINING_EPOCHS=3
BATCH_SIZE=1
```

## 3. Build Docker Images

### Main API + Worker

```bash
docker-compose build
```

### Training Service (GPU)

```bash
docker-compose --profile training build trainer
```

### Inference Service (GPU)

```bash
docker-compose --profile inference build inference
```

## 4. Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi
```

Kết quả phải hiển thị GPU của bạn.

## 5. Start Services

### API Server

```bash
docker-compose up -d
```

API available at: http://localhost:8000

### Training (xem TRAINING.md)

```bash
docker-compose --profile training up trainer
```

### Inference API

```bash
docker-compose --profile inference up -d inference
```

Inference API at: http://localhost:8001

## 6. Test Installation

```bash
# Test API
curl http://localhost:8000/health

# Test GPU in container
docker run --rm --gpus all ielts-trainer python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Docker Desktop GPU không hoạt động

1. Bật WSL2 integration trong Docker Desktop Settings
2. Settings → Resources → WSL Integration → Enable
3. Restart Docker Desktop

### Build lỗi

```bash
# Clean build
docker-compose down -v
docker system prune -a
docker-compose build --no-cache
```

### Port đã được sử dụng

Chỉnh sửa ports trong `docker-compose.yml`:

```yaml
ports:
  - "8080:8000" # thay vì 8000:8000
```
