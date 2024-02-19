# 山倉隆太卒業研究

卒業研究で作成したベクター画像生成AIのコードです。またこのコードは、<a href="https://github.com/Shiriluz/Word-As-Image">Word-As-Image</a>をもとに改変し作成しました。
なお、コードの清書は未だ行われていません。あくまで成果物として評価をいただけると幸いです。
概要に関しては概要.pdfをご覧ください。




## Setup
セットアップは、<a href="https://github.com/Shiriluz/Word-As-Image">Word-As-Image</a>に従って行ってください。
1. Clone the repo:
```bash
git clone https://github.com/WordAsImage/Word-As-Image.git
cd Word-As-Image
```
2. Create a new conda environment and install the libraries:
```bash
conda create --name word python=3.8.15
conda activate word
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
conda install -y numpy scikit-image
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils numba torch-tools scikit-fmm easydict visdom freetype-py shapely
pip install opencv-python==4.5.4.60  
pip install kornia==0.6.8
pip install wandb
pip install shapely
```

3. Install diffusers:
```bash
pip install diffusers==0.8
pip install transformers scipy ftfy accelerate
```
4. Install diffvg:
```bash
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
```

5. Paste your HuggingFace [access token](https://huggingface.co/settings/tokens) for StableDiffusion in the TOKEN file.
## Run Experiments 
```bash
conda activate word
cd Word-As-Image

# Please modify the parameters accordingly in the file and run:
bash run_word_as_image.sh

```
* ```--semantic_concept``` : the semantic concept to insert
* ```--optimized_letter``` : one letter in the word to optimize
* ```--font``` : font name, the <font name>.ttf file should be located in code/data/fonts/

Optional arguments:
* ```--word``` : The text to work on, default: the semantic concept
* ```--config``` : Path to config file, default: code/config/base.yaml
* ```--experiment``` : You can specify any experiment in the config file, default: conformal_0.5_dist_pixel_100_kernel201
* ```--log_dir``` : Default: output folder
* ```--prompt_suffix``` : Default: "minimal flat 2d vector. lineal color. trending on artstation"

### Examples
```bash
python code/main.py  --semantic_concept "bunny" --target_file "star"
```
<br>
