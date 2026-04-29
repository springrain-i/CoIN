# Claude 记忆迁移包

迁移自 `/root/.claude/projects/-root/memory/`，日期：2026-04-28

## 包含内容

- `memory/MEMORY.md` — 记忆索引
- `memory/project_coin.md` — CoIN / MoE-MoKA 项目状态
- `memory/project_moe_moka_bugs.md` — Bug 修复记录 + 验证结果
- `memory/user_profile.md` — 用户背景
- `memory/reference_hytmp.md` — /hy-tmp 路径参考

## 新服务器恢复步骤

```bash
# 1. 克隆仓库（如果还没有）
git clone git@github.com:springrain-i/CoIN.git
cd CoIN
git checkout MoEMoKA

# 2. 确定新服务器的 Claude 记忆路径
#    Claude Code 的全局记忆默认在 ~/.claude/projects/ 下
#    路径格式：~/.claude/projects/<encoded-workdir>/memory/
#    在 /root/CoIN 工作目录下，路径为：
MEMORY_DIR="$HOME/.claude/projects/-root-CoIN/memory"
# 注意：路径 encoding 可能因服务器不同而变化，
# 建议先打开 Claude Code 在 /root/CoIN 目录，让它自动创建路径，再复制

# 3. 复制记忆文件
mkdir -p "$MEMORY_DIR"
cp docs/claude-migration/memory/*.md "$MEMORY_DIR/"

# 4. 验证
ls "$MEMORY_DIR"
```

## 注意事项

- **不含** `.credentials.json`（API 密钥），需在新服务器重新登录 `claude` CLI
- **不含** `settings.json`（权限配置），Claude Code 会在新服务器重新生成
- **不含** hooks/skills/commands（属于全局 Claude Scholar 配置，不是 CoIN 项目专属）
- 大模型文件（Vicuna、CLIP）需确认新服务器路径，更新 `coin_paths.sh` 中的路径变量

## 关键路径对照（当前服务器）

| 资源 | 路径 |
|------|------|
| Base model | `/hy-tmp/Vicuna/vicuna-7b-v1.5` |
| Vision tower | `/hy-tmp/clip-vit-large-patch14-336` |
| 训练数据 | `/hy-tmp/playground/Instructions_Original/` |
| 图像数据 | `/hy-tmp/` |
| Checkpoint 输出 | `/root/CoIN/checkpoints/LLaVA/CoIN/` |
| Conda 环境 | `coin`（Python 3.10） |
