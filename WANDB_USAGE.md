# Wandb 日志集成使用说明

## 概述

本项目已集成 Weights & Biases (wandb) 日志系统，可以将训练过程中的所有统计信息和控制台消息自动上传到 wandb，方便远程监控和分析实验结果。

## 功能特性

### 1. 自动记录统计数据
所有通过 `logger.log_stat()` 记录的统计指标都会自动同步到 wandb，包括：
- 训练损失（td_error_abs, q_taken_mean, target_mean）
- 回报值（episode_return）
- 选择结果（select_sp_return）
- 其他自定义指标

### 2. 捕获控制台消息
所有通过 `logger.console_logger.info()` 打印的消息都会自动发送到 wandb，包括：
- 训练进度信息
- 时间估计
- 模型保存通知
- 各种训练阶段的状态信息

### 3. 统计摘要
`logger.print_recent_stats()` 打印的统计摘要也会同步到 wandb。

## 安装 wandb

```bash
pip install wandb
```

首次使用需要登录：
```bash
wandb login
```

## 使用方法

### 方法 1：修改配置文件

编辑 `config/default.yaml`，设置：
```yaml
use_wandb: True
wandb_project: "EvoSARL"  # 项目名称
wandb_name: ""  # 运行名称（留空则使用 unique_token）
```

### 方法 2：命令行参数

运行时通过命令行参数启用：
```bash
python main.py --use_wandb=True --wandb_project="MyProject" --wandb_name="experiment_1"
```

### 示例

```bash
# 基本用法
python main.py --env-config=sc2 --config=qmix --use_wandb=True

# 指定项目和运行名称
python main.py \
  --env-config=sc2 \
  --config=qmix \
  --use_wandb=True \
  --wandb_project="EvoSARL-Experiments" \
  --wandb_name="qmix-sc2-test"
```

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_wandb` | bool | False | 是否启用 wandb 日志 |
| `wandb_project` | str | "EvoSARL" | wandb 项目名称 |
| `wandb_name` | str | "" | 运行名称（空则使用 unique_token） |

## 实现细节

### 代码修改

1. **utils/logging.py**
   - 添加了 `WandbHandler` 类，用于捕获控制台日志
   - 在 `Logger` 类中添加了 `setup_wandb()` 方法
   - 修改了 `log_stat()` 方法，添加 wandb 日志支持

2. **run.py**
   - 在 logger 初始化后添加了 `setup_wandb()` 调用
   - 自动传递配置参数到 wandb

3. **config/default.yaml**
   - 添加了 wandb 相关配置项

### 日志流程

```
训练过程
    ↓
logger.log_stat(key, value, t)
    ├─→ TensorBoard（如果启用）
    ├─→ Sacred（如果启用）
    └─→ Wandb（如果启用）

logger.console_logger.info(message)
    ├─→ 控制台输出
    └─→ Wandb（通过 WandbHandler）
```

## 注意事项

1. **依赖安装**：使用前需要安装 wandb：`pip install wandb`
2. **网络连接**：wandb 需要网络连接来上传数据
3. **登录认证**：首次使用需要运行 `wandb login` 进行认证
4. **兼容性**：wandb 功能不会影响现有的 TensorBoard 和 Sacred 日志系统
5. **评估模式**：在评估模式（`evaluate=True`）下，wandb 不会被启用

## 故障排除

### wandb 未安装
如果 wandb 未安装，系统会输出警告但不会中断程序：
```
WARNING: wandb not installed. Install it with: pip install wandb
```

### 初始化失败
如果 wandb 初始化失败，系统会输出警告：
```
WARNING: Failed to setup wandb: <error message>
```

### 查看日志
训练开始时，如果 wandb 成功启用，会看到：
```
INFO: Wandb logging enabled. Project: <project_name>
```

## 查看结果

访问 https://wandb.ai 查看你的实验结果，包括：
- 实时训练曲线
- 控制台日志
- 配置参数
- 系统信息

## 禁用 wandb

如果不需要 wandb 日志，只需保持 `use_wandb: False`（默认值）即可。
