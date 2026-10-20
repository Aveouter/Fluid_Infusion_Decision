## 尚待核实（提交前必须完成）

- [ ] 用师兄最终修订稿中的**最终标题**替换占位符；核对标题、关键词不含缩写，并与投稿系统一致。
- [ ] 拿到最终修订稿和 Response Letter 后，逐段确认下方英文信中的每项修改都已真实出现在终稿，并与逐条回复一致；补齐稿件中的图表/章节定位。
- [ ] 仓库目前没有 `docs/response_matrix.md`。取得最终 response matrix（或确认实际采用的逐条回复文件）后，核对各项修订摘要与证据及终稿落点。
- [ ] 确认作者姓名与顺序、通讯作者姓名和联系方式。
- [ ] 请全体作者确认原创、未一稿多投、作者同意及利益冲突声明；按确认结果替换文末声明，不能未经确认直接保留。
- [ ] 核对修订后的指标和结论。当前证据不支持“始终优于基线”：患者级配对结果中，MAE 差异区间跨零，且 persistence 的患者级 macro-F1 更高；Cover Letter 不应宣称模型优越。
- [ ] 终稿核对后生成上传用 PDF，请 @Aveouter 复核；通过后再记录终稿路径与 commit，并按 #94/#28 的流程完成提交记录。

---

## Cover Letter (English draft)

**Internal working draft — do not submit until all items above are checked and all bracketed fields are replaced.**

October 20, 2026

Editor-in-Chief  
*Engineering Applications of Artificial Intelligence*

**Re: Revised Manuscript EAAI-26-19194**

Dear Editor-in-Chief,

We respectfully resubmit our revised manuscript, “[INSERT THE FINAL TITLE FROM THE APPROVED REVISED MANUSCRIPT]” (manuscript number **EAAI-26-19194**), for consideration as a **Revised Manuscript** in *Engineering Applications of Artificial Intelligence*. An accompanying Response Letter provides our point-by-point replies to the reviewers.

The revision clarifies that the task is to predict infusion types and rates recorded in the clinical data, and distinguishes one-step prediction using observed history from recursive rollout. It reports persistence and moving-average baselines and patient-level paired uncertainty estimates, together with calibration, subgroup, and failure-case analyses. The comparisons do not establish consistent superiority over simple baselines; the conclusions have therefore been revised to reflect the observed results and their uncertainty.

We audited the methods against the released implementation. The manuscript now describes the implemented hybrid temporal-convolutional and attention-based architecture and its one-timestep input, and corrects the account of the classification and regression objectives. It no longer presents unused Tversky, focal, or temporal-regularization components as implemented, and clarifies that the zero-mass component was not fitted using the stated zero-inflated log-normal likelihood. The preprocessing description has also been corrected to disclose the use of each patient's full 72-hour series for imputation and scaling, rather than claiming train-only fitting.

The manuscript’s interpretation is limited to agreement with recorded clinical practice. It does not claim that the model identifies optimal treatment, improves patient outcomes, or provides a clinically superior treatment strategy. The methods, results, discussion, tables, and figures have been revised to keep the claims within the available evidence. **Confirm this paragraph against the final revised manuscript and Response Letter before submission.**

[CONFIRM: This work is original, has not been published previously, and is not under consideration by another journal.]  
[CONFIRM: All authors have approved this revised submission.]  
[CONFIRM: The authors have no conflicts of interest, or replace with the approved disclosure.]

Sincerely,

[CORRESPONDING AUTHOR NAME]  
[Institution]  
[Postal address]  
[Email address]  
[Telephone, if required]
