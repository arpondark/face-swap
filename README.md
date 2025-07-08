## Note (If you want to install on docker use roop.rar file)
First extract the rar file then continue .
A research and educational tool for face swapping technology, forked from the original ROOP project

> **Note**: This repository complies with GitHub's Terms of Service and Acceptable Use Policies. The software is provided for legitimate research, educational, and creative purposes only.

### Features

- Cross-platform Browser-based GUI
- Advanced face detection and processing capabilities
- Multiple processing modes and configurations
- Batch processing for images and videos
- AI-powered face enhancement and restoration
- Real-time preview functionality
- Modular architecture with plugin support
- Comprehensive settings and configuration management
- Modern UI with theme support

*For research, education, and legitimate creative applications only*


## ⚠️ Important Disclaimer

**ETHICAL USE ONLY**: This software is intended exclusively for:
- Academic research and education
- Legitimate creative and artistic projects  
- Technical demonstrations and learning
- Authorized commercial applications with proper consent

**PROHIBITED USES**: 
- Creating non-consensual content
- Impersonation or identity theft
- Harassment, defamation, or malicious use
- Any illegal activities under your local jurisdiction

**USER RESPONSIBILITY**: 
- Users must obtain explicit consent before processing anyone's likeness
- Content must be clearly labeled as AI-generated when shared
- Users are solely responsible for compliance with local laws and regulations
- Developers assume no liability for misuse of this technology

**By using this software, you agree to use it responsibly and ethically.**

## Installation Guide


---

## 🪟 Windows Installation

### Prerequisites
Before starting, ensure you have the following installed:

1. **Python 3.8-3.11** (NOT 3.12+): [Download from Python.org](https://www.python.org/downloads/)
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH" during installation
   
2. **Git**: [Download from Git-SCM](https://git-scm.com/download/win)

3. **Visual Studio Build Tools 2019/2022**: [Download here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Install "C++ build tools" workload

4. **FFmpeg**: [Download from FFmpeg.org](https://ffmpeg.org/download.html#build-windows)
   - Extract and add to PATH, or use: `winget install FFmpeg`

### For NVIDIA GPU Users (Recommended)

#### Step 1: Install NVIDIA Drivers
1. Download latest NVIDIA drivers: [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Install CUDA Toolkit 11.8: [CUDA 11.8 Download](https://developer.nvidia.com/cuda-11-8-0-download-archive)

#### Step 2: Install Miniconda (Recommended)
```powershell
# Download and install Miniconda
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "miniconda.exe"
.\miniconda.exe /InstallationType=JustMe /RegisterPython=1 /S /D=%UserProfile%\Miniconda3
```

#### Step 3: Setup Project (NVIDIA RTX 30/40 Series)
```powershell
# Clone repository
git clone https://github.com/arpondark/face-swap.git
cd face-swap

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install CUDA toolkit
conda install -c nvidia cudatoolkit=11.8 -y

# Install requirements
pip install -r requirements.txt
pip install "numpy<2.0"
pip install --upgrade gradio==5.13.0 fastapi pydantic

# Run the application
python run.py
```

#### Step 4: Setup Project (NVIDIA RTX 50 Series - Latest)
```powershell
# Clone repository
git clone https://github.com/arpondark/face-swap.git
cd face-swap

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install UV for faster package management
pip install uv

# Install PyTorch with CUDA 12.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install core packages
uv pip install numpy opencv-python-headless onnx insightface albucore psutil
uv pip install onnxruntime-gpu tqdm ftfy regex pyvirtualcam

# Install specific versions
pip install --force-reinstall pydantic==2.10.6
pip install --upgrade gradio==5.13.0

# Run the application
python run.py
```

### For AMD GPU Users
```powershell
# Clone and setup
git clone https://github.com/arpondark/face-swap.git
cd face-swap
python -m venv venv
.\venv\Scripts\activate

# Install DirectML for AMD
pip install onnxruntime-directml
pip install -r requirements.txt
pip install "numpy<2.0"
pip install --upgrade gradio==5.13.0 fastapi pydantic

# Run the application
python run.py
```

### For CPU Only (No GPU)
```powershell
# Clone and setup
git clone https://github.com/arpondark/face-swap.git
cd face-swap
python -m venv venv
.\venv\Scripts\activate

# Install CPU version
pip install onnxruntime
pip install -r requirements.txt
pip install "numpy<2.0"
pip install --upgrade gradio==5.13.0 fastapi pydantic

# Run the application
python run.py
```

---

## 🐧 Linux Installation (Ubuntu/Debian)

### Linux Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv git build-essential cmake
sudo apt install -y ffmpeg libsm6 libxext6 libfontconfig1 libxrender1

# Install Python 3.8-3.11 if not available
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

### For NVIDIA GPU Users

#### Step 1: Install NVIDIA Drivers
```bash
# Check if NVIDIA GPU is detected
lspci | grep -i nvidia

# Install NVIDIA drivers (Ubuntu)
sudo apt install -y nvidia-driver-535  # or latest version
# OR use the graphics drivers PPA for latest drivers
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install -y nvidia-driver-545  # latest version

# Reboot after driver installation
sudo reboot
```

#### Step 2: Install CUDA Toolkit
```bash
# Download and install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvidia-smi
nvcc --version
```

#### Step 3: Install Miniconda (Optional but Recommended)
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

#### Step 4: Setup Project
```bash
# Clone repository
git clone https://github.com/arpondark/face-swap.git
cd face-swap

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install CUDA toolkit (if using conda)
conda install -c nvidia cudatoolkit=11.8 -y

# Install requirements
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu
pip install "numpy<2.0"
pip install --upgrade gradio==5.13.0 fastapi pydantic

# Run the application
python run.py
```

### For AMD GPU Users (ROCm)
```bash
# Install ROCm (AMD GPU support)
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7 ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install -y rocm-dev rocm-libs

# Setup project
git clone https://github.com/arpondark/face-swap.git
cd face-swap
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
pip install onnxruntime
pip install "numpy<2.0"
pip install --upgrade gradio==5.13.0 fastapi pydantic

python run.py
```

---

## 🔧 WSL (Windows Subsystem for Linux) with NVIDIA GPU

### Step 1: Install WSL2
```powershell
# Run in Windows PowerShell as Administrator
wsl --install -d Ubuntu-22.04
# Reboot when prompted
```

### Step 2: Install NVIDIA Drivers for WSL
1. **On Windows Host**: Install latest NVIDIA drivers that support WSL: [NVIDIA WSL Drivers](https://developer.nvidia.com/cuda/wsl)
2. **Do NOT install NVIDIA drivers inside WSL** - use the Windows host drivers

### Step 3: Setup CUDA in WSL
```bash
# Inside WSL Ubuntu
# Remove any existing CUDA installations
sudo apt-get purge nvidia* cuda*

# Install CUDA Toolkit for WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# Verify GPU access in WSL
nvidia-smi
```

### Step 4: Install Prerequisites in WSL
```bash
# Update and install essentials
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git build-essential cmake
sudo apt install -y ffmpeg libsm6 libxext6 libfontconfig1 libxrender1

# Install Python 3.10
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

### Step 5: Setup Roop-Floyd in WSL
```bash
# Clone repository
git clone https://github.com/arpondark/face-swap.git
cd face-swap

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu
pip install "numpy<2.0"
pip install --upgrade gradio==5.13.0 fastapi pydantic

# Test GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run the application
python run.py
```

### WSL Troubleshooting
If you encounter issues:

1. **GPU not detected**: Ensure Windows NVIDIA drivers support WSL and are up to date
2. **CUDA errors**: Verify CUDA installation with `nvcc --version`
3. **Memory issues**: Increase WSL memory limit in `.wslconfig` file:
```ini
# In Windows: %UserProfile%\.wslconfig
[wsl2]
memory=8GB
processors=4
```

---

## 🐳 Docker Installation

Docker provides an isolated environment for running Roop-Floyd with all dependencies pre-configured. This is the easiest way to get started regardless of your operating system.

### Prerequisites for Docker

1. **Docker Desktop**: [Download for Windows](https://docs.docker.com/desktop/install/windows-install/) | [Download for Linux](https://docs.docker.com/desktop/install/linux-install/)
2. **For GPU Support**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Quick Start with Docker

#### Option 1: CPU-Only Version (No GPU Required)

```bash
# Clone the repository
git clone https://github.com/arpondark/face-swap.git
cd face-swap

# Create required directories
mkdir -p input output models temp

# Build and run with Docker Compose
docker-compose up --build
```

#### Option 2: GPU-Accelerated Version (NVIDIA Only)

**Step 1: Install NVIDIA Docker Support**

For Ubuntu/Debian:
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

For Windows:
- Ensure you have WSL2 with GPU support enabled
- Install NVIDIA drivers for WSL
- Docker Desktop will automatically detect GPU support

**Step 2: Run GPU Version**

```bash
# Clone the repository
git clone https://github.com/arpondark/face-swap.git
cd face-swap

# Create required directories
mkdir -p input output models temp

# Build and run GPU version
docker-compose -f docker-compose.gpu.yml up --build
```

### Manual Docker Commands

#### CPU Version
```bash
# Build the image
docker build -t face-swap:cpu .

# Run the container
docker run -d \
  --name face-swap-app \
  -p 7860:7860 \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  face-swap:cpu
```

#### GPU Version
```bash
# Build the GPU image
docker build -f Dockerfile.gpu -t face-swap:gpu .

# Run with GPU support
docker run -d \
  --name face-swap-gpu-app \
  --gpus all \
  -p 7860:7860 \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  face-swap:gpu
```

### Docker Environment Configuration

#### Volume Mapping
- `./input:/app/input` - Place your source images/videos here
- `./output:/app/output` - Processed results will appear here
- `./models:/app/models` - AI models will be stored here
- `./temp:/app/temp` - Temporary processing files

#### Environment Variables
You can customize the Docker container with environment variables:

```yaml
environment:
  - GRADIO_SERVER_NAME=0.0.0.0  # Server host
  - GRADIO_SERVER_PORT=7860     # Server port
  - CUDA_VISIBLE_DEVICES=0      # GPU device (GPU version only)
```

### Production Deployment

For production deployment, you can use the included reverse proxy configuration:

1. **Enable NGINX in docker-compose.yml**:
```yaml
# Uncomment the nginx service in docker-compose.yml
```

2. **Create SSL certificates** (optional):
```bash
mkdir ssl
# Add your SSL certificates to the ssl/ directory
```

3. **Deploy**:
```bash
docker-compose up -d
```

### Docker Troubleshooting

#### Common Issues:

1. **Permission Denied on Volume Mounts**:
```bash
# Fix permissions
sudo chown -R 1000:1000 input output models temp
```

2. **GPU Not Detected**:
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi
```

3. **Out of Memory**:
```bash
# Limit memory usage
docker run --memory="4g" --name face-swap-app face-swap:cpu
```

4. **Port Already in Use**:
```bash
# Use different port
docker run -p 8080:7860 face-swap:cpu
```

### Docker Updates

To update to the latest version:

```bash
# Pull latest code
git pull origin main

# Rebuild containers
docker-compose down
docker-compose up --build
```

### Accessing the Application

Once the Docker container is running:

1. Open your web browser
2. Navigate to: `http://localhost:7860`
3. Upload your files using the web interface
4. Configure settings and start processing!

### Docker Performance Tips

- **Use GPU version** for significantly faster processing
- **Allocate sufficient RAM** (8GB+ recommended)
- **Use SSD storage** for volume mounts
- **Close other containers** to free up resources
- **Monitor resource usage** with `docker stats`

---

## 🚀 Usage

### Windows
```powershell
# Activate virtual environment
.\venv\Scripts\activate
# Run application
python run.py
```

### Linux/WSL
```bash
# Activate virtual environment
source venv/bin/activate
# Run application
python run.py
```

### Access the Interface
1. Open your web browser
2. Navigate to: `http://localhost:7860`
3. Upload your source and target images/videos
4. Configure settings and start face swapping!

---

## 📋 Google Colab Installation

For easy cloud-based usage:

1. Download `face-swap-colab.ipynb`
2. Upload to Google Colab
3. Run all cells in sequence
4. Access the interface through the provided Gradio link

---

## ⚠️ Troubleshooting

### Common Issues:

1. **ImportError**: Install missing packages with `pip install <package-name>`
2. **CUDA errors**: Ensure CUDA version matches PyTorch requirements
3. **Memory issues**: Reduce batch size or use CPU mode
4. **FFmpeg errors**: Ensure FFmpeg is installed and in PATH

### Performance Tips:

- Use NVIDIA GPU for best performance
- Close other GPU-intensive applications
- Ensure adequate RAM (8GB+ recommended)
- Use SSD storage for faster processing

---

## 📄 License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

**Remember**: Always use this software responsibly and in compliance with your local laws and GitHub's terms of service.




