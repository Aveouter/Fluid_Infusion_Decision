# NM-v1 复现说明

本目录是一次完整实验轮次的落点。以下是按顺序重跑它所需的全部命令、产物与验收点。所有命令在仓库根目录执行，Python 解释器为 `./.venv/Scripts/python.exe`（Windows）或 `python`（POSIX）。

## 环境

- Python 3.12，`requirements.txt` 与 `requirements-dev.txt` 中的依赖；训练需要 PyTorch（本轮在 CUDA 上运行，CPU 亦可，速度约慢一个数量级）。
- 数据入口固定为 `docs/dual_task_review_protocol_v4.json` 的 `feature_file`，其 SHA256 在协议中钉死；`utils/review_data_context.py:load_features` 在读取时校验。

## 步骤

```bash
# 0. 一次性：登记协议（训练前）
#    docs/nm_protocol.json 已包含目标、选择规则与统计口径

# 1. 开发集五折 × 三种子 × 全部变体（可断点续跑，已存在 prediction.npz 即跳过）
python run_nm.py
#    单变体：python run_nm.py --variant main --seeds 3

# 2. 内层 OOF 阈值与护栏选择（只读内层预测，绝不看外折结果）
python reports/nm_select.py --variant main --seeds 3
python reports/nm_select.py --variant no_summary --seeds 1     # 其余变体同理

# 3. 开发集评估：主模型、全部变体、全部基线、配对区间、分层误差、失败病例
python evaluate_nm.py

# 4. 审计：因果性、患者隔离、缺失、库存守恒、预测文件一致性、权重重载
python reports/nm_audit.py

# 5. 增强溯源：强度、随机种子、前后分布、被改动的通道
python reports/nm_augmentation_provenance.py

# 6. 冻结阈值与权重，然后才允许触碰历史留出集
python finalize_nm.py fit
python finalize_nm.py holdout

# 7. 冻结引擎端点（P1 累计剂量 / P2 容差 / P3 分类等）
python reports/nm_engine_endpoints.py

# 8. 打包：唯一写入 docs/experiments/nm-v1-20261015/ 的脚本，已存在即拒绝覆盖
python reports/package_nm_study.py

# 9. 测试
python -m pytest -q
```

## 关键产物

| 路径 | 内容 |
| --- | --- |
| `<run>/<variant>/fold<k>/inner<j>_seed<s>/prediction.npz` | 内层 OOF 的逐窗口头输出 |
| `<run>/<variant>/fold<k>/outer_seed<s>/` | 外折逐窗口预测 + `bundle.pt` 权重 + `scope.json` 患者范围 |
| `<run>/<variant>/fold<k>/selection.json` | 该折的事件阈值与护栏参数（仅由内层 OOF 决定） |
| `<run>/evaluation/` | 开发集汇总、分液体、配对区间、分层误差、消融、种子离散度、失败病例 |
| `<run>/holdout/` | 留出集结果与冻结引擎端点 |
| `<run>/audit.json` | Gate A 与 Gate G 的检查结果 |
| `<run>/freeze_before_holdout.json` | 冻结时的源码哈希、协议与阈值 |

## 复现检查点

- `run_nm.py` 幂等：重复执行不会重新训练已完成的折。
- `finalize_nm.py holdout` 在 `freeze_before_holdout.json` 缺失或源码哈希变动时**拒绝运行**。
- `reports/nm_audit.py` 的 `passed` 必须为真，否则不得进入正式比较。
- `reports/package_nm_study.py` 拒绝覆盖已发布的聚合包。
