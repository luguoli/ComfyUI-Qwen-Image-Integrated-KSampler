# 🐋 ComfyUI 千问图像集成采样器（Qwen Image Integrated KSampler）

[![GitHub](https://img.shields.io/badge/GitHub-luguoli-orange)](https://github.com/luguoli)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-自定义节点-blue)](https://github.com/comfyanonymous/ComfyUI)

[English](README-en.md) | **简体中文**

QwenImageIntegratedKSampler

这是一个集成化的ComfyUI Qwen-Image 生图的采样器节点，支持Z-Image，相比使用官方的K采样器，告别原版乱七八糟的连线，同时支持文生图和图生图生成，解决了官方节点生图偏移的问题，并且集成了提示词输入框、图片自动缩放、显存/内存自动自动清理、批量生图、自动保存等全面优化功能，妈妈再也不用担心我连线乱七八糟了~~~~

#### 如果这个项目对您有帮助，请点个 ⭐Star 吧，让我知道世界上还有人类在使用它！

## 🏆 特色功能

### 🎨 支持生成模式
- **Z-Image**: 支持Z-Image模型
- **文生图**: 从文本提示生成图像
- **图生图**: 基于参考图生成、图片编辑，支持5张图像

### ⚡ 高级优化
- **优化偏移问题**: 解决了官方节点生图偏移的问题，且能更好的遵从指令
- **采样算法（AuraFlow）集成**: 集成了采样算法（AuraFlow）节点，无需另外连线
- **CFGNorm集成**: 集成了CFGNorm节点，无需另外连线

### 🖼️ 图像处理
- **提示词输入框集成**: 集成了提示词输入框，无需另外连线
- **多张参考图像**: 支持最多5张参考图像进行条件生成
- **自动图像缩放**: 在调整到目标尺寸的同时保持纵横比
- **支持ControlNet控制**: 可额外连接[千问 ControlNet 集成加载器]进行姿态、深度等控制

### 🔧 提高生产力
- **批量生成**: 在单次操作中生成多张图像
- **自动显存清理**: GPU/VRAM显存的自动清理选项
- **自动内存清理**: RAM内存的自动清理选项
- **自动保存结果**: 自动将生成的结果图像保存到指定文件夹
- **完成声音通知**: 生成完成后播放音频提醒


## 🍧 对比展示 
### 🔄 工作流复杂度对比
- **❌ 未使用【千问图像集成采样器】的工作流（复杂繁琐，超多节点，超多连线）**
![alt text](images/1-1.png)
- **✅ 使用了【千问图像集成采样器】的工作流（极简，单节点搞定，几乎无连线）**
![alt text](images/1-2.png)

### 🖼️ 生成图像效果对比
- **❌ 未使用【千问图像集成采样器】的工作流（明显偏移、缩放）**
![alt text](images/2-1.png)
- **✅ 使用了【千问图像集成采样器】的工作流（完全无偏移、缩放）**
![alt text](images/2-2.png)


## 📦 安装方法

### 方法1: 通过ComfyUI管理器（推荐）
1. 在ComfyUI界面中打开ComfyUI管理器
2. 搜索 "ComfyUI-Qwen-Image-Integrated-KSampler"
3. 点击安装

### 方法2: 手动安装
1. 导航到您的ComfyUI自定义节点目录：
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   ```

2. 克隆仓库：
   ```bash
   git clone https://github.com/luguoli/ComfyUI-Qwen-Image-Integrated-KSampler.git
   或 Gitee 仓库：
   git clone https://gitee.com/luguoli/ComfyUI-Qwen-Image-Integrated-KSampler.git
   ```

3. 安装依赖项：
   ```bash
   pip install -r requirements.txt
   ```

4. 重启ComfyUI

## 🚀 使用方法

### [工作流示例](workflow_example.json)


### 基础文生图生成

1. 将"🐋 千问图像集成采样器"节点添加到工作流中
2. 设置 `generation_mode` 为 "文生图"
3. 连接必需输入：
   - 模型 (🤖 Model)
   - CLIP (🟡 Clip)
   - VAE (🎨 Vae)
4. 输入正向和负向提示词
5. 设置宽度和高度（文生图必填）
6. 配置采样参数（步数、CFG、采样器、调度器）
7. 执行工作流

### 图生图生成

1. 将节点添加到工作流中
2. 设置 `generation_mode` 为 "图生图"
3. 至少连接一张参考图像（🖼️ Image1）
4. 可选添加最多4张其他参考图像
5. 输入正向/负向提示词和指令
6. 设置目标宽度/高度用于缩放（可选）
7. 根据需要配置其他参数
8. 执行工作流

### ControlNet 控制

1. 添加[千问 ControlNet 集成加载器]节点，连线至[ControlNet 数据]
2. 连接姿态、深度控制图
3. 选择ControlNet模型，设置控制类型和强度
4. 执行工作流
![alt text](images/3.png)


### 高级功能

- **内存管理**: 启用GPU/CPU清理选项以提高资源效率
- **批量处理**: 设置 batch_size > 1 进行多张图像生成
- **自动保存**: 指定输出文件夹进行自动保存
- **AuraFlow调优**: 调整 auraflow_shift 以平衡速度与质量
- **CFG增强**: CFG 的稳定器

## ⚠️ 注意事项

### 📝 使用要求
- **文生图模式**：必须设置宽度（Width）和高度（Height），这是必填参数
- **图生图模式**：必须至少提供一张参考图像（Image1），最多支持5张参考图像（Image1-Image5）

### 🎛️ 参数设置建议
- **批量大小（Batch Size）**：1-10之间选择，根据GPU内存调整，建议从1开始测试
- **分辨率（Width/Height）**：须为8的倍数，范围0-16384，建议从较低分辨率（如512x512）开始测试
- **采样步数（Steps）**：千问模型推荐4-20步，过高可能增加计算时间但不一定提升质量
- **CFG值**：0-100范围，默认1.0，推荐1.0-7.0范围
- **降噪强度（Denoise）**：0-1范围，默认1.0，图生图模式可适当降低
- **AuraFlow Shift**：0-100范围，默认3.0，用于平衡生成速度与质量
- **CFG规范化强度**：0-100范围，默认1.0，CFG 的稳定器

### 🔧 图像处理
- **自动缩放**：文生图必须输入宽高参数，图生图填写自动缩放参考图像保持纵横比，宽高任意填0则不缩放
- **参考图像顺序**：最多支持5张参考图像，按Image1-Image5优先级处理，Image1为主图
- **图像格式**：支持标准图像输入格式，自动处理批次维度

### 💾 内存管理
- **GPU内存清理**：启用enable_clean_gpu_memory选项，在生成前/后自动清理VRAM
- **CPU内存清理**：启用enable_clean_cpu_memory_after_finish，生成结束后清理RAM（包含文件缓存、进程、动态库）
- 连续大量生成时建议始终启用内存清理选项以防止内存溢出

### 💾 自动保存
- **输出文件夹**：设置auto_save_output_folder启用自动保存功能，置空则不自动保存，支持绝对路径和相对路径
- **文件命名**：output_filename_prefix自定义前缀，默认"auto_save"
- 保存格式为PNG，文件名包含种子和批次编号（如：auto_save_123456_00000.png）

### 🔊 通知功能
- **声音通知**：仅在Windows系统支持


## 📝 更新记录
### v1.0.6：
- **增加汉化脚本：** ComfyUI从v0.3.68开始中文语言文件失效，增加自动汉化脚本，双击执行【自动汉化节点.bat】后重启ComfyUI即可，需要安装ComfyUI-DD-Translation插件


## 📞 需要特别定制请联系 📞 
- 作者：@luguoli（墙上的向日葵）
- 作者邮箱：luguoli@vip.qq.com


---

**用❤️为ComfyUI社区制作**
