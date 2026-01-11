<div align="center">

## CharaConsist: Fine-Grained Consistent Character Generation

Official implementation of ICCV 2025 paper - CharaConsist: Fine-Grained Consistent Character Generation

[[Paper](https://arxiv.org/abs/2507.11533)] &emsp; [[Project Page](https://murray-wang.github.io/CharaConsist/)] &emsp; <br>
</div>

## Update
----
## Branches
- origin: 原始工作代码，我们添加了的一个快速简易运行的脚本run_inference.sh, 和可视化报告所述点匹配失败的脚本point.sh
- modified: 报告所述改进后的代码，当前效果尚有欠缺，持续更新中
- adain：报告中讨论部分提及，曾尝试过风格引导来约束一致性的尝试，但是现在效果看上去这种train-free方法对风格捕捉较差，暂时放弃。
- sam-mask：报告中讨论部分提及，曾尝试过利用sam进行mask分割，但是认为这个mask在早期步就需要，sam识别不到足够的语义信息，此分支跑出来可视化效果很差，暂时放弃。
----


## How to use
### Dependencies and Installation
需要：
- CUDA support
- PyTorch >= 2.0.0
- diffusers
 
提供以下两种环境配置方式，此两种都是复现时自己重新配置环境导出，可以直接使用：
```bash
conda env create -f environment.yml
```

```bash
conda create --name characonsist python=3.9
conda activate characonsist
pip install -U pip
# Install requirements
pip install -r requirements.txt
```
### Pretrained Model
本项目依赖预训练的FLUX.1-dev模型，请先自行下载好该模型，官方路径https://huggingface.co/black-forest-labs/FLUX.1-dev


### Quick Start
我们提供了三种使用CharaConsist生成一致性角色的方式：

#### (1) 快速脚本（推荐新手使用）
最简单的入门方式是使用我们自行编写的 `run_inference.sh` 脚本，它为常见用例提供了预配置设置。

**基本用法：**
```bash
# 首先激活conda环境
conda activate characonsist

# 运行推理脚本
bash run_inference.sh
```

**脚本功能：**
- 使用预配置提示生成一致性角色
- 自动设置多GPU配置以获得更好的性能
- 自行选择保存结果路径
- 包含mask可视化用于分析

**`run_inference.sh` 中的可调节参数：**
```bash
# 硬件设置
INIT_MODE=2                    # 0: 单GPU (37GB), 1: CPU卸载 (26GB), 2: 多GPU (≤20GB), 3: 顺序卸载 (3GB)
GPU_IDS="0 1"                 # 要使用的GPU ID
MODEL_PATH="/path/to/FLUX.1-dev"  # FLUX.1-dev模型路径

# 输出设置
OUT_DIR="results/two_people_v4"   # 输出目录

# 图像设置
HEIGHT=1024                    # 图像高度
WIDTH=1024                     # 图像宽度
SEED=2025                      # 随机种子，保证可重现性

# 内容设置（你可以修改这些！）
BG_PROMPTS=(                   # 背景描述
    "In a sunny park with green grass and blooming flowers,"
    "In a peaceful park with a small pond and willow trees,"
    "In a park playground with colorful equipment and benches,"
)
FG_PROMPT="a little girl with pigtails, wearing a yellow dress, and a little boy with cap, wearing shorts and a t-shirt,"  # 角色描述
ACT_PROMPTS=(                  # 动作描述
    "running with a kite, laughing, full body view"
    "sitting on a bench, eating ice cream, happy expression, front view"
    "playing on a swing, excited expression, dynamic view"
)
```

#### (2) Notebook用于单个示例
我们提供了三个Jupyter notebooks用于详细探索：
- `gen-bg_fg.ipynb`: 在固定背景下生成一致性角色
- `gen-fg_only.ipynb`: 在不同背景下生成一致性角色
- `gen-mix.ipynb`: 在部分固定和部分变化的背景下生成相同角色

用户可以参考notebooks中的详细描述来熟悉方法的整个框架。

#### (3) 脚本用于批量生成
对于需要精细控制的高级用户，可以直接使用 `inference.py` 脚本。

**关键参数：**

**硬件配置：**
- `init_mode`: 内存优化策略
    | init_mode | 初始化方式 | GPU内存 | GPU数量 |
    |-----------|-----------|---------|---------|
    | 0 | 单GPU | 37 GB | 1 |
    | 1 | 单GPU + CPU卸载 | 26 GB | 1 |
    | 2 | 多GPU分布式 | ≤20 GB | ≥2 |
    | 3 | 顺序CPU卸载 | 3 GB | 1 |
- `gpu_ids`: 要使用的GPU设备ID

**内容控制：**
- `prompts_file`: 提示配置文件路径
- `model_path`: FLUX.1-dev模型权重路径
- `out_dir`: 输出目录
- `height/width`: 图像分辨率
- `seed`: 随机种子

**功能开关：**
- `use_interpolate`: 启用自适应token合并（提高一致性，但增加CPU内存消耗）
- `share_bg`: 保持背景在各帧中不变
- `save_mask`: 保存生成过程中自动提取的mask用于可视化
- `save_point_match`: 保存点匹配数据
- `visualize_masks`: 启用matplotlib mask可视化
- `save_visualizations`: 保存mask可视化到文件
- `visualize_denoise_steps`: 在特定去噪步骤可视化mask（例如："10,20,30"）

**使用示例：**

固定背景生成：
```bash
python inference.py \
--init_mode 0 \
--prompts_file examples/prompts-bg_fg.txt \
--model_path path/to/FLUX.1-dev \
--out_dir results/bg_fg \
--use_interpolate --save_mask --share_bg
```

跨背景生成：
```bash
python inference.py \
--init_mode 0 \
--prompts_file examples/prompts-fg_only.txt \
--model_path path/to/FLUX.1-dev \
--out_dir results/fg_only \
--use_interpolate --save_mask
```
我们补充了逐步可视化的内容，便于观察生成效果

高级可视化（去噪步骤）：
```bash
python inference.py \
--init_mode 0 \
--prompts_file examples/prompts-fg_only.txt \
--model_path path/to/FLUX.1-dev \
--out_dir results/fg_only \
--use_interpolate --save_mask \
--visualize_denoise_steps "10,20,30,40,50"
```


