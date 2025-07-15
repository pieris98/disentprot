#!/bin/bash
# AWS setup script for protein disentanglement experiments

set -e

echo "Setting up AWS environment for protein disentanglement experiments..."

# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git htop tree

# Install NVIDIA drivers and CUDA (for GPU instances)
if lspci | grep -i nvidia; then
    echo "Installing NVIDIA drivers and CUDA..."
    sudo apt-get install -y nvidia-driver-470
    
    # Install CUDA toolkit
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda
    
    echo "CUDA installed. Please reboot the instance."
fi

# Create project directory
PROJECT_DIR="$HOME/protein-disentanglement"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Clone repository (assuming you'll push to GitHub)
# git clone https://github.com/yourusername/protein-disentanglement.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data experiments logs

# Download sample protein data (optional)
echo "Setting up sample protein data..."
python3 -c "
from src.data.utils import create_sample_fasta
import os
os.makedirs('data', exist_ok=True)
create_sample_fasta('data/sample_proteins.fasta', num_samples=1000)
print('Sample FASTA created with 1000 sequences')
"

# Set up AWS CLI (if not already configured)
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    pip install awscli
fi

# Create systemd service for long-running experiments (optional)
cat > protein-disentanglement.service << EOF
[Unit]
Description=Protein Disentanglement Experiments
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python train.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "Setup complete!"
echo "==================="
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Configure AWS credentials: aws configure"
echo "3. Test installation: python train.py --debug"
echo "4. Run experiments: python train.py --model all"
echo ""
echo "Useful commands:"
echo "- Monitor GPU: nvidia-smi"
echo "- Monitor training: tail -f logs/training.log"
echo "- Start experiment service: sudo systemctl start protein-disentanglement"
