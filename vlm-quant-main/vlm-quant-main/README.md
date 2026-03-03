# vlm-quant with nvfp4
本项目主要实现使用nvfp4量化qwen2.5-vl模型并进行推理测试、精度测试和速度测试。项目基于fp-quant、qutlass、transformers、lmms-eval、TensorRT Model Optimizer等构建。
## 环境配置
**建议使用示例的docker从一个干净的环境配置使用**
1. 启动docker
```shell
docker run --runtime=nvidia -v /ssd/wja/:/root/wja --shm-size 32g --network host  -itd --name=wja2-docker nvcr.io/nvidia/pytorch:25.04-py3 /bin/bas
h
```
2. 配置初始环境
```shell
apt update
apt upgrade
apt install tmux git wget vim -y
# 下面两步是为了避免docker本身的constrain.txt限制
mv /etc/pip/constraint.txt /etc/pip/constraint.txt.bk
touch /etc/pip/constraint.txt
vim .bashrc #配置conda环境变量和代理
```
3. 配置conda环境和pytorch
```shell
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n vlm-quant python=3.12 -y
conda activate vlm-quant
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```
4. 安装配置其余仓库     
本仓库依赖其余的仓库，例如qutlass、fast-hadmard-transform、flash-attention等。
```shell
cd third_party
cd fp-quant
pip install -e .
cd qutlass
pip install --no-build-isolation -e .
cd transformers
pip install -e .
cd lmms-eval
pip install -e .
cd fast-hadamard-transform
pip install -v .
pip install flash-attn --no-build-isolation
```
## 运行使用
- 运行量化
```shell
./run_vlm.sh
```
- 测试量化后的模型推理
```shell
python src/test_infer_vlm.py
```
- 测试模型精度
这里使用lmms-eval仓库进行精度测试，由于该仓库不支持直接的测试量化模型精度，因此需要修改对应的源文件进行量化模型的测试。
请参考vlm-quant/third_party/lmms-eval/lmms_eval/models/simple/qwen2_5_vl.py中的Qwen2_5_VL的init部分的内容修改对应路径（line 113）
然后可以使用官方脚本进行测试。
```shell
bash third_party/lmms-eval/examples/models/qwen25vl.sh
```
- 测试速度
```shell
python benchmark/benchmark_vlm_single_layer.py
python benchmark/benchmark_linear.py
```