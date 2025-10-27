conda create -n test python=3.10 -y
conda activate test
# 安裝
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
