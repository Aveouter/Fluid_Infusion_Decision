# CI/CD

本仓库的 CI 只使用合成 fixture，不读取 `data/` 下的临床数据。CI 能看到的只有
**已跟踪文件**；未提交的测试和模块不会运行。

## 门禁（PR 与 main）

| 检查名 | 命令 | 作用 |
| --- | --- | --- |
| `lint` | `python -m ruff check .` | 语法、未定义名字、`__all__` 破损、赋值前使用 |
| `protocol-freeze` | `python freeze_eval_protocol.py` | 评估协议文档/代码/JSON 不漂移 |
| `repo-hygiene` | `python tools/check_repo_hygiene.py` | 禁止数据文件、患者编号形态、常见密钥 |
| `test` | `python -m pytest` | 合成 fixture 测试；`gpu`/`realdata` 自动跳过 |

四个 job 并行运行；`test` 在失败时仍上传 `junit.xml`。Actions 全部按 commit SHA
固定，Dependabot 每周更新。

`ruff` 只开 `E9/F63/F821/F822/F823`，是“能不能跑”的门禁，不是风格门禁。要加规则
先在本地跑一遍并评估存量噪声，不要直接放宽到 `E/F` 全开。

## 本地复现

```bash
# 与 CI 相同的 CPU 环境
pip install -r requirements-ci.txt

python -m ruff check .
python freeze_eval_protocol.py
python tools/check_repo_hygiene.py
python -m pytest
```

本地有 GPU 时用 `requirements-dev.txt`；两者只差 torch 的轮子：

| 文件 | torch | 用途 |
| --- | --- | --- |
| `requirements-base.txt` | 无 | 公共精确 pin |
| `requirements.txt` | `2.14.1+cu126` | 本地 CUDA |
| `requirements-ci.txt` | `2.14.1+cpu` | CI / release |
| `requirements-dev.txt` | 继承 `requirements.txt` | 本地开发 |
| `requirements-lint.txt` | 无 | lint job |

`tests/conftest.py` 在没有 CUDA 时跳过 `gpu`，在没有 `data/baseline.pkl` 时跳过
`realdata`。CI 的设置使这两类分支每次都会走到——这正是
`tests/test_skip_paths.py` 要防的“本地绿、CI 红”问题。

## 发布流程

版本号约定 `vMAJOR.MINOR.PATCH`。release workflow 必须在被 tag 的 commit 上已经存在。

```bash
git switch main
git pull
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```

tag push 会触发 `.github/workflows/release.yml`：

1. 校验 tag 格式、tag 指向的 commit、以及该 commit 在 `main` 上；
2. 在该 commit 上重跑 lint、协议、hygiene、pytest；
3. 生成 `artifacts/release-manifest.json`（commit/tree、依赖版本、协议 hash、
   artifact SHA-256）；
4. 创建 GitHub Release，附 `release-manifest.json`、`junit.xml`、
   `docs/eval_protocol.json`、`docs/eval_protocol.md`。

预演（不创建 release、不需要 tag）：

```bash
gh workflow run release.yml --ref main -F dry_run=true
```

重复发布会被拒绝（`gh release view` 已存在即失败）。需要重跑时先删除：

```bash
gh release delete v0.1.0 --yes
```

## 分支保护

`main` 要求：

- 必须通过 PR 合并；
- `lint`、`protocol-freeze`、`repo-hygiene`、`test` 四个检查必须通过；
- 分支必须与 `main` 同步（strict）；
- 禁止强推与删除。

`enforce_admins: true` 表示管理员也受约束；紧急情况下可在仓库设置里临时关闭。
如果重命名了 job，必须同步更新 branch protection 的 contexts。

## 仓库卫生策略

`tools/check_repo_hygiene.py` 扫描 `git ls-files` 的当前树，发现以下内容即失败：

- 数据/产物后缀：`.pkl .joblib .csv .xlsx .xls .npy .npz .parquet .onnx .pt .pth
  .h5 .ckpt .zip .tar .gz .pem .key`；
- 目录：`data/`、`ckpts/`、`reports/`、`preds/`、`ml_output*/`、
  `per_patient_outputs*/`；
- `.env` / `.env.*`（放行 `.env.example`）；
- 常见密钥模式（AWS、GitHub、Slack、私钥头、Google、OpenAI）；
- `patient_id` / `subject_id` / `mrn` / `pid` 后 40 字符内出现 6 位以上数字。

初期不设 allowlist。确实需要提交小聚合表时，先审查内容，再用 `git add -f`；如果
误报，修改脚本里的显式豁免并写清理由，不要直接跳过整个检查。

## 已知限制

- 只扫描当前已跟踪树，**不扫描 git 历史**。历史中的旧数据是独立问题，见
  `docs/ci_cd.md` 的“后续”一节。
- manifest 不是签名的，也不锁定依赖 hash；它记录的是实际安装版本和 commit。
- CI 只覆盖已跟踪测试。当前工作区里还有大量未跟踪测试，需另开 PR 分批提交。
- Dependabot 忽略 `torch`，因为 `+cu126` / `+cpu` 本地标签必须与索引一起升级。

## 后续

1. **未跟踪测试套件**：135 个测试模块 + 163 个 `utils` 模块 + 协议 JSON 需分批提交。
   提交前给已知无守卫的真实数据测试补 `realdata` 标记或 skip。
2. **公开历史隐私**：`origin` 历史仍可取得 2025 年删除的逐患者 CSV/PNG。CI 只能
   防新的提交；需要单独决定历史重写（`git filter-repo` + force-push + 联系 GitHub
   Support 清理 + 处理 fork）或把仓库转为私有。默认假设这些数据已经被下载过。
3. 可选：Windows 冒烟 job、coverage、依赖 hash 锁、签名 manifest。
