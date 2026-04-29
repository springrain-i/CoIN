---
name: /hy-tmp 访问方式
description: 说明 /hy-tmp 为何在当前 Claude Code 会话中可访问，以及跨会话访问的注意事项
type: reference
originSessionId: 1332c3fb-e149-47b9-87b5-362104822c56
---
`/hy-tmp` 是服务器上的一个共享存储卷（shared storage volume），挂载在当前运行 Claude Code 会话的容器/节点里。

**可访问的条件：**
- 从同一个挂载了 `/hy-tmp` 的终端/容器启动 Claude Code 会话
- 当前会话的工作环境（`/root`）与 `/hy-tmp` 处于同一挂载命名空间

**不可访问的原因（另一个 session）：**
- 不同 SSH 节点、不同容器、或不同挂载环境启动的会话看不到该挂载点
- 这是底层系统/容器隔离问题，与 Claude 无关

**实际使用约定：**
- 大型 checkpoint（模型权重、数据集等）一律存在 `/hy-tmp`，不放 `/root`（`/root` 磁盘空间有限）
- 代码和脚本放 `/root/MokA/`
- 关键路径：
  - Llama-2-7B-chat: `/hy-tmp/data/home/sqx/MokA/VisualText/checkpoints/llama-2-7b-chat-hf`
  - CLIP ViT: `/hy-tmp/clip-vit-large-patch14-336`
  - CoIN 数据: `/hy-tmp/playground/Instructions_Original`
  - 图像根目录: `/hy-tmp`
  - visual_pretrain.bin（目标位置）: `/hy-tmp/checkpoints/visual_projector/visual_pretrain.bin`
