# Storyboard Project

A storyboard generation system that integrates multiple advanced image and video generation technologies.

## Installation

### Requirements
- Python 3.10+
- CUDA-compatible GPU (recommended)
- At least 24GB GPU memory

### Installation Steps

```bash
# Create virtual environment
conda create -n storyboard python=3.10 -y
conda activate storyboard

# Install PyTorch (CUDA version)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 (if want to use grounding dino.)

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Start the server**:
   ```bash
   python storyboard_server.py
   ```

2. **Access the web interface**:
   - Open your browser and visit `http://localhost:7860`

3. **Create storyboards**:
   - Upload images to nodes
   - Set transition descriptions
   - Generate video sequences



## Model Licenses

This project integrates multiple third-party models, each distributed under its respective license terms.  
Users are responsible for complying with the original licenses when using or redistributing these components.

| Model Name | License Type | Commercial Use | Source Repository / Reference |
|-------------|---------------|----------------|--------------------------------|
| Qwen2.5-VL-7B-Instruct | Apache License 2.0 *(verify specific version; some releases may include additional restrictions)* | Permitted | [https://github.com/QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) |
| Segment Anything Model (SAM) | Apache License 2.0 | Permitted | [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) |
| FramePack | Apache License 2.0 | Permitted | [https://github.com/lllyasviel/FramePack](https://github.com/lllyasviel/FramePack) |
| ObjectClear | NTU S-Lab License 1.0 *(Non-Commercial)* | Not permitted | [https://github.com/zjx0101/ObjectClear](https://github.com/zjx0101/ObjectClear) |
| Qwen-Image-Edit | Apache License 2.0 | Permitted | [https://huggingface.co/Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) |
| Qwen-Image-Edit2509 | Apache License 2.0 | Permitted | Base: Qwen-Image-Edit — [https://huggingface.co/Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) |



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
