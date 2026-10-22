## 尚待核实（提交前必须完成）

- [ ] 用师兄最终修订稿中的**最终标题**替换占位符；核对标题、关键词不含缩写，并与投稿系统一致。
- [ ] 逐项核对下面标记为“仅在终稿确认后保留”的内容；若修订稿没有完成对应修改，删除或改写该段。
- [ ] 仓库默认分支目前没有 `docs/response_matrix.md`，也未提供完整 Response Letter。取得最终回复材料后，核实信中每项修改摘要都能对应到实际修订稿和逐条回复；不得把 PR #99 的单条回复当作完整回复信。
- [ ] 确认作者姓名与顺序、通讯作者姓名和联系方式。
- [ ] 请全体作者确认原创、未一稿多投、作者同意及利益冲突声明；按确认结果替换文末声明，不能未经确认直接保留。
- [ ] 核对修订后的指标和结论。当前证据不支持“始终优于基线”：患者级配对结果中，MAE 差异区间跨零，且 persistence 的患者级 macro-F1 更高；Cover Letter 不应宣称模型优越。
- [ ] 终稿核对后生成上传用 PDF，请 @Aveouter 复核；通过后再记录终稿路径与 commit，并按 #94/#28 的流程完成提交记录。

---

## Cover Letter (English draft)

**Internal working draft only. Do not submit until all bracketed checks and author confirmations are complete.**

October 22, 2026

Editor-in-Chief  
*Engineering Applications of Artificial Intelligence*

**Re: Revised Manuscript EAAI-26-19194**

Dear Editor-in-Chief,

We respectfully resubmit our revised manuscript, “[INSERT THE FINAL TITLE FROM THE APPROVED REVISED MANUSCRIPT]” (manuscript number **EAAI-26-19194**), for consideration as a **Revised Manuscript** in *Engineering Applications of Artificial Intelligence*. [CONFIRM that the complete point-by-point Response Letter is included with the submission; PR #99 addresses only Reviewer 1, Comment 1 and is not the complete Response Letter.]

[RETAIN ONLY AFTER CHECKING THE APPROVED MANUSCRIPT:] The revised manuscript clarifies that the prediction targets are infusion types and rates recorded in the clinical data and distinguishes one-step prediction from recursive rollout. It reports persistence and moving-average baselines, patient-level paired uncertainty estimates, calibration, subgroup, and failure-case analyses. The patient-level comparisons do not establish consistent superiority over simple baselines, and the conclusions have been limited accordingly. [VERIFY every item and the final reported values against the approved manuscript and response letter.]

[RETAIN ONLY AFTER CHECKING THE APPROVED MANUSCRIPT:] The methods description has been audited against the released implementation. The revision describes the implemented hybrid temporal-convolutional and attention-based architecture and its one-timestep input; corrects the classification and regression objectives; removes claims about unused Tversky, focal, and temporal-regularization components; clarifies that the zero-mass component was not fitted using the stated zero-inflated log-normal likelihood; and accurately describes preprocessing using each patient's full 72-hour series for imputation and scaling. [VERIFY these changes are present in the approved manuscript.]

[RETAIN ONLY AFTER CHECKING THE APPROVED MANUSCRIPT:] The manuscript frames the work as an evaluation of agreement with recorded clinical practice. It does not claim that the model identifies optimal treatment, improves patient outcomes, or provides a clinically superior treatment strategy. [VERIFY the final text preserves these limits.]

[AUTHOR CONFIRMATION REQUIRED: This work is original, has not been published previously, and is not under consideration by another journal.]  
[AUTHOR CONFIRMATION REQUIRED: All authors have approved this revised submission.]  
[AUTHOR CONFIRMATION REQUIRED: State the approved conflict-of-interest disclosure.]

Sincerely,

[CORRESPONDING AUTHOR NAME]  
[Institution]  
[Postal address]  
[Email address]  
[Telephone, if required]
