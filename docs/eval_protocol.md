# 评估协议 v2（冻结）

> 本文件是 BurnResu 修订版的**唯一评估协议**。冻结常量区块由
> `freeze_eval_protocol.py` 从 `utils/eval_protocol.py` 生成；叙事部分手写。
> 校验：`python freeze_eval_protocol.py`（不一致即非零退出）。

**适用范围。** 本协议只规定"如何计算与报告"，不规定模型结构、损失函数或
训练流程。评估引擎只接受**已解码的预测值**，不知道预测来自 TCN、GRU、
Transformer、ZILN 还是持久性基线。

---

## 0. 两条必须先说清的声明

### 0.1 原稿 §3.6 没有预设累计剂量的时域

原稿 §3.6 的原文是 "cumulative-dose error measured by MAE and tolerance
accuracy at ±200 mL and ±10%"，**未指定具体时域**；72 小时是数据集与任务
窗口的定义（§3.2 与图 5 注），不是"原预设的主要终点时域"。

因此本协议**不追溯性**宣称 72h 为原始主要终点。协议把时域写成显式的
`horizon_hours` 字段，并同时报告 24/48/72 三个时域各自的 `n`，使读者可以
自行判断，而不依赖任何事后叙事。原稿 §3.6 明确写出的主要分类指标
（Macro-F1）与次要指标清单（AUC、precision、recall、instantaneous MAE、
RMSE、positive-flow SMAPE、change-point timing、temporal stability）在本
协议中逐项对应实现。

### 0.2 回归类终点的主口径尚未冻结

`speed`、`rate`、`action_rate_uncond`、`action_rate_cond` 等口径**语义已
冻结**（§2），但**主口径的选择标为 `proposed`，本版不冻结**。本轮把全部
候选口径作为独立、明确标识的评估输入并行实现；主解码策略须在正式比较前
另行预先指定，且**不得依据测试结果选择最优解码**。

---

## 1. 引擎接口

引擎只接受已解码的预测值：

```python
evaluate(pred: PredictionSet, *, protocol, quality_mask=None,
         model_id, run_id, split_or_outer_fold, train_seed, level="target")
```

`PredictionSet` 携带 `patient_ids`、`time_index`、`true_bits`、`true_rates`、
`pred_rates`（**已解码**）、可选的 `pred_bits`/`pred_prob`，以及
`rate_definition` 与 `eval_mode` 两个身份字段。

前置校验不通过即拒绝运行：

- `bit = 1 ⟺ rate > 0`（标签协议的核心不变量）；
- 同一患者的 `time_index` 连续且有序，不得跨 split/fold；
- `rate_definition`、`eval_mode` 在词表内；
- 结果行的 `protocol_hash` 与冻结产物一致。

**确定性解码**（`1[prob ≥ τ_c] × speed` 等）不属于指标定义，单独放在
`utils/decoding.py`，作为冻结的 `DECODERS` 词表。解码器是纯函数，指标代码
不含任何解码逻辑。

---

## 2. 速率口径的语义（按源码写死）

模型的实际输出链路：`pos_mean = exp(μ + σ²/2) − 1`（**条件**正流均值），
`flow_pred = (1 − p0_ZILN) · pos_mean`（**无条件**均值 E[Y]），
`speed = softplus(flow_pred) ≈ flow_pred`。

| 名称 | 定义 | 真实语义 |
|---|---|---|
| `pos_mean` | `exp(μ + σ²/2) − 1` | **条件正流均值**（对数正态部分） |
| `flow_pred` | `(1 − p0_ZILN) · pos_mean` | ZILN **无条件**均值 E[Y] |
| `speed` | `softplus(flow_pred) ≈ flow_pred` | **无条件**均值经 softplus。**不是条件正流均值** |
| `rate` | `prob_cls × speed` | 分类头概率**再次**缩放无条件均值——门控了两次 |
| `action_rate_uncond` | `1[prob_cls ≥ τ_c] × speed` | 阈值门控后的动作解码（无条件均值版） |
| `action_rate_cond` | `1[prob_cls ≥ τ_c] × pos_mean` | 动作解码（条件正流强度版） |
| `deterministic` | `bit = 1 ⟺ rate > 0` | 无概率头的基线（持久性、移动平均） |

**结构性说明。** `prob_cls` 来自分类头，`p0_ZILN` 是 ZILN 自己的零质量
logit，二者不是同一个"是否输注"概率。`rate = prob_cls × speed` 因此把两个
不同模型的概率混在一起相乘。该结构性缺陷归模型工单处理；本协议只在
此处记录事实，不回填、不解释成"条件速率"。

**阈值来源。** 分类阈值必须来自验证集并记录来源。禁止静默回退到 0.5；
无来源即输出 `NA + na_reason`。

**实现现状。** `pos_mean` 与 `flow_pred` 目前不在预测产物中，需要模型侧
产者。在本轮中它们输出 `NA + na_reason="producer_not_available"`，
**不得**从 `speed` 反推。

---

## 3. 估计对象与时间算术

| ID | `estimand` | 定义 | `eval_mode` |
|---|---|---|---|
| **E1** | `conditional_cumulative_volume` | `V̂(T) = V₀ + Σ r̂_t`，`V₀` 为决策时刻**已观测**的首小时（非泄漏），求和项为以真实历史为条件的一步预测 | `one_step_observed_history` |
| **E2** | `action_autoregressive_rollout_under_true_initial_state` | 以真实初始标签与真实初始累计量起步，之后由模型自身输出递推。**不得称 fully closed-loop** | `action_autoregressive_rollout` |
| **E2b** | `observed_physiology_conditioned_rollout` | 使用未来真实生理特征时升格为此名 | 同上 |
| **E3** | `pure_prediction_window_volume` | `Σ r̂_t`，无观测锚点 | 同 E1 |

三条必须同时成立的事实：

1. **E1 与 E3 的终末累计绝对误差必然相同**：
   `|(V₀ + V̂_S) − (V₀ + V_S)| = |V̂_S − V_S|`。
   因此 E3 **不能**作为 E1 误差的独立稳健性证据；E3 只在**相对容差分母、
   累计量绝对水平、时域解释**三处提供信息。
2. 72 个**记录**小时只对应 **71 个模型预测小时**（首小时是观测锚点）。
   24/48/72 个记录小时分别只用 23/47/71 个预测小时。
3. E3 必须标为 **23/47/71 小时纯预测窗**，不得在表中写成 24/48/72。

同一结果文件可存放多个 estimand（每行带 `estimand` 身份），但**禁止**在
统计汇总或配对比较中跨 estimand 混用——由 `fairness_check()` 拦截。

---

## 4. 结果行的身份

每个数字必须携带完整身份，规范键由下列字段拼接：

`model_id | run_id | level | metric_name | aggregation | unit |
horizon_hours | horizon_origin | estimand | fluid_class | mask |
eval_mode | rate_definition | split_or_outer_fold | train_seed |
threshold | quality_track | protocol_hash`

- `model_id`/`run_id` 不可省略：否则持久性与移动平均在同患者集合、同时域
  下都用 `deterministic`，键会完全撞上。
- `quality_track` 区分主表与敏感性表；`protocol_hash` 保证定义变更后新旧
  结果不会被当成同一个东西比较。
- 展平导出时**冲突即抛错**，不允许静默覆盖。

**缺失的表达。** `value = None` 当且仅当 `na_reason` 非空；非空值必须是
有限数。禁止写出裸 `NaN`（不是合法 JSON）。

**真实零值 / 缺失 / 结构不完整必须分开**：三者使用不同 `mask`
（`zero_flow` / `missing` / `incomplete`），不得合并，也不得把不可估计
写成 0。不可估计一律 `NA + na_reason`。

---

## 5. 八条硬性门禁

| # | 门禁 |
|---|---|
| 1 | **P1**：患者在各自窗内先累积，再算绝对误差，最后**患者等权平均**。合成测试必须区分终末累计 **150 mL** 与 all-hours 患者 MAE **50 mL/h** 这两个不同量纲的终点 |
| 2 | **P2**：±200 mL 与 ±10% **分别报告**为两个独立主终点；零剂量患者**不进入相对容差的分母**，单列桶计数与零剂量准确率 |
| 3 | **P3**：固定三个液体类别的 Macro-F1 与 `zero_division=0` 规则；`present_class_macro_f1` 仅作敏感性；ROC-AUC 的单类标签缺失仍记 NA（规则不同，理由写明） |
| 4 | **事件匹配**：最大基数一对一匹配优先、其次最小化总时间偏移；**不得**使用最近邻贪心 |
| 5 | **时域**：72 记录小时 = 71 预测小时；严格完整性校验，不足即 `NA + insufficient_horizon` |
| 6 | **统计**：独立单位是患者；F1/AUC 的 bootstrap 必须**重采样后重建标签与预测再算**，不得先求每人的 F1 再平均 |
| 7 | **结果身份**：模型、run、mask、horizon、estimand、口径、协议版本齐全 |
| 8 | **缺失处理**：真实零值、缺失、结构不完整分别处理；不可估计返回 `NA + reason` |

---

## 6. 统计契约

- **独立性单位是患者**，不是小时、不是窗口。同一患者的全部小时必须落在
  同一 split/fold。
- 外层设计：5-fold grouped CV + 嵌套验证调参；关键深度模型至少 3 个固定
  训练种子。若因算力/数据约束缩减，**必须在查看新测试结果之前**记录缩减
  内容与理由。
- **两类不确定性分开标识**：跨 fold/seed 的均值±SD（训练不确定性）与同一
  评估队列上的患者 bootstrap 95% CI（抽样不确定性）。两者不得混算，也不
  得互相代替。
- **配对比较**：两个方法共享同一组重采样患者索引；报告 ΔMAE、ΔF1、
  Δ累计剂量 MAE 的 95% CI。区间跨 0 即标记"未建立优势"。AUC 使用保留患者
  簇的重采样，**不得**对相关小时套用普通独立样本 DeLong。
- **预先声明主比较与多重比较解读规则**；不因结果不利而删除终点、阈值或基线。
- **模型选择规则先冻结**：优先优化累计剂量误差；Macro-F1 与变化时点误差
  设为不退化约束。
- **基线公平性**：全部方法共用同一患者 manifest、目标、有效样本索引、缺失
  处理与同一个 exporter；非序列模型必须给显式历史滞后特征以对齐可见信息；
  逐模型记录超参搜索预算；输入差异**记录而非假称相同**（例如基线没有概率
  头，其口径是 `deterministic`）。
- **诚实标注**：原 15% 测试集已被报告过。基于它的评估属于**回顾性单次
  评估**，不是嵌套的，逐行标注「非嵌套；测试集已被看过」，**不得**作为
  泛化证据。

---

## 7. 双轨真值与可估计性

- **族群与小时网格在主表与敏感性表两轨中完全相同**。质量掩码只改变小时的
  合格标记，**绝不增删患者**。
- 计算敏感性表之前先算患者交集，并报告各时域下的有效人数。**不得**把质量
  掩码数据集的规模直接当作模型测试集的分母。
- 主表资格规则：患者在时域 T 合格 ⟺ T 个小时全部存在且标签有效。
- 敏感性表资格规则：患者在时域 T 合格 ⟺ T 个小时全部 `analysis_eligible`。
- 若某时域在严格掩码下合格人数为 0，**明确宣布该终点在该轨下不可估计**
  （`NA + na_reason="not_estimable_under_quality_mask"`），**不得**放宽条件
  后仍称其为同一终点。
- 每张表附 `population_delta` 审计行，报出因掩码转为不合格的患者数与小时数。

---

## 8. 医学层（§2A）的证据边界

- **C1（早期复苏暴露强度）**：`V_cryst(0–24h) / (weight_kg × TBSA_percent)`，
  单位 mL/kg/%TBSA。体重与 TBSA 取自基线特征；缺失即 `NA + 原因`，并报出
  不可计算的比例。**指南的 2 mL/kg/%TBSA 是起始估算建议，不是个体金标准
  目标**，只能作为 `guidance_only` 行报告，不得用作达标率的分母。
- **C2（分液体剂量组成忠实度）**：逐液体累计误差、总误差，以及分量误差和
  `Σ_c |V̂_ic − V_ic|`，用于识别"总量一致但液体种类相互抵消"的失败。
- **N1 与 PS-TC 必须分开**：`titration_concordance` 用**可用**的分层（如小时
  段、观测速率水平）计算描述性调速一致性；**PS-TC 要求真实生理状态分层**
  （低尿量、血流动力学），在生理状态不可得时输出 `NA + 原因`。**不得**用
  时间段冒充生理状态。
- **N2（CETE）**：累计暴露轨迹误差，按 24/48/72h 与三类液体分别报告，同时
  给出按体重归一化（mL/kg）与未归一化（mL）两个版本。
- N1/N2 标注 `proposed_exploratory`（本研究拟议、未经临床验证），**不得**
  取代 P1/P2/P3，也不得凭其最高值宣称医学优越性。
- **C-C 类结局**（AKI、肾替代治疗、ARDS/肺水肿、腹腔间隔室综合征、死亡、
  ICU 住院天数）：在当前可用数据中均**不可评估**，一律 `not assessable`
  并说明缺失原因，**不生成虚假医学分数**。

---

## 9. 冻结常量

<!-- BEGIN GENERATED:PROTOCOL-CONSTANTS -->

**协议版本**：`2.0`　　**冻结哈希**：`849a59c17baf8815`

> 哈希由本区块的全部冻结项经 sha256 计算（前 16 位）。任何阈值或词表变更都会改变哈希；冻结后修改定义必须递增版本并保留变更记录。

**容差与阈值**

| 项 | 值 |
|---|---|
| `TOL_ABS_ML` | 200.0 |
| `TOL_PCT` | 0.1 |
| `CHANGE_ABS_ML` | 50.0 |
| `CHANGE_REL` | 0.25 |
| `REL_BASE_ML` | 50.0 |
| `EPS_RATE_ML_PER_H` | 0.0 |
| `MATCH_TOLERANCE_PRIMARY` | 1 |
| `MATCH_TOLERANCE_SENSITIVITY` | 2 |

**时域**

| 项 | 值 |
|---|---|
| `HORIZONS` | [24, 48, 72] |
| `预测小时数@24 记录小时` | 23 |
| `预测小时数@48 记录小时` | 47 |
| `预测小时数@72 记录小时` | 71 |

**统计契约**

| 项 | 值 |
|---|---|
| `N_OUTER_FOLDS` | 5 |
| `N_SEEDS_MIN` | 3 |
| `N_BOOTSTRAP` | 2000 |
| `CI_LEVEL` | 0.95 |
| `INDEPENDENCE_UNIT` | patient |

**词表**

- `eval_mode`： `one_step_observed_history`、`action_autoregressive_rollout`、`na`
- `rate_definition`： `pos_mean`、`flow_pred`、`speed`、`rate`、`action_rate_uncond`、`action_rate_cond`、`deterministic`、`na`
- `estimand`： `conditional_cumulative_volume`、`action_autoregressive_rollout_under_true_initial_state`、`observed_physiology_conditioned_rollout`、`pure_prediction_window_volume`、`na`
- `horizon_origin`： `hour_0_anchored`、`first_predicted_hour`、`na`
- `mask`： `all`、`positive_flow`、`zero_flow`、`missing`、`incomplete`、`true_change_hour`、`no_change_hour`、`zero_dose_patient`、`eligible_patient`、`matched_events`、`na`
- `fluid_class`： `total`、`crystal`、`colloid`、`water`、`na`
- `quality_track`： `main`、`adjudicated_mask`、`na`
- `evidence_tier`： `measured`、`derived`、`proposed_exploratory`、`guidance_only`、`not_assessable`、`na`
- `observation_source`： `observed`、`imputed`、`unavailable`、`na`
- `unit`： `probability`、`count`、`ratio`、`hours`、`mL/h`、`mL`、`mL/kg`、`mL/kg/%TBSA`
- `aggregation`： `per_class`、`macro`、`micro`、`weighted`、`samples`、`patient_mean`、`patient_median`、`patient_share`、`target_mean`、`target_pooled`、`event_macro`、`event_micro`
- `level`： `window`、`patient`

**解码规则**： `threshold_gate_uncond`、`threshold_gate_cond`

<!-- END GENERATED:PROTOCOL-CONSTANTS -->

---

## 10. 已知限制

1. `pos_mean` / `flow_pred` 需要模型侧产者；本轮缺失即 NA，**不得从 `speed` 反推**。
2. PS-TC 的生理分层不可得：建模产物中的尿量是 z-score 且按体重归一化后的值，
   不是 mL/kg/h；无 MAP、无乳酸、无碱剩余。
3. 目标侧缺失率不可计算：`-1` 哨兵在 `utils/label_protocol.normalize` 中被
   置 0，**未记录与真实零输注不可区分**。该语义只在合成 fixture 中演示。
4. C1 的"伤后 0–24h"与标签第 0 个记录小时是否重合，无法从公开仓库核实。
5. 阳性率接近 0/1 时 bootstrap 百分位区间会退化；Wilson 区间作为敏感性输出。
6. 分类阈值的验证集来源无法从公开仓库核实；真实运行断言其与本地阈值文件
   一致并记录指纹。
7. 本轮为**回顾性单次评估**，非嵌套、原测试集已被看过；泛化证据需后续的
   患者级嵌套交叉验证。
8. `prob_cls`（分类头）与 `p0_ZILN`（ZILN 零质量）不是同一个概率，
   `rate = prob_cls × speed` 门控两次——该结构性缺陷归模型工单。
