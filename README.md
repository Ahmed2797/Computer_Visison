sudo apt install python3.12
python3.12 -m venv cv
source cv/bin/activate   


sudo apt update
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
 

source ~/.bashrc
conda --version

conda create -n cv python=3.12 -y
conda activate cv 


pip install numpy pandas opencv-python ultralytics
pip install ultralytics --index-url https://pypi.org/simple 

$ pip install -r requirements.txt




conda env list        # List environments
conda remove -n cv --all   # Delete environment

pip install -r requirements.txt 

# YOLO
https://github.com/ultralytics/ultralytics 


