
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from taskweave.types.structured_result import StructuredResult, TaskStatus
from taskweave.types.task_graph import FailureCategory


class WorkspaceFinalState(str, Enum):
    """Worktree 在 task 退出时的最终状态。

    这是系统观察的核心字段：用户分支真的拿到产物的唯一证据是
    `MERGED`。其他状态都意味着"产物没回到主干"。
    """

    NEVER_CREATED = "never_created"
    """worktree 从未创建（依赖失败 / 创建异常）"""

    CREATED_BUT_FAILED = "created_but_failed"
    """worktree 建了但任务执行失败"""

    COMMITTED_NOT_MERGED = "committed_not_merged"
    """已提交到 subagent 分支但未合并到主干（中间任务 / 冲突）"""

    MERGED = "merged"
    """本任务的 worktree 直接 merge 进了用户分支 — leaf 节点的成功路径"""

    MERGED_VIA_DESCENDANT = "merged_via_descendant"
    """本任务自己未被独立 merge，但其 commit 已通过下游 leaf 的 merge 进入用户分支。

    DAG 中间节点（被下游依赖的任务）走这条路径：
      - leaf 任务的 worktree 基于其依赖的 subagent 分支创建（git 父子关系）
      - leaf merge 到 main 时，整条 ancestor 链上的 commit 一并进入 main
    所以"中间节点产物有没有落地"只取决于"它是否有任意一条下游 leaf 成功 merge"。
    DAG 收尾阶段会回溯 ancestors 把 COMMITTED_NOT_MERGED 升级为本状态。"""

    CORRUPTED = "corrupted"
    """状态不一致（内存 vs 磁盘 vs git 漂移），需人工介入"""

    CLEANED_NO_PRODUCT = "cleaned_no_product"
    """worktree 被清理但没有产物落地（agent 没写文件 / commit 失败）"""

    NOT_APPLICABLE = "not_applicable"
    """该角色不需要 worktree（Planner / Searcher）"""


@dataclass
class AgentReport:
    """Agent 自报。**参考意见**，不可信任为事实。

    所有字段都是从 submit_task_result 工具调用里提取的原始内容，
    系统对这些值不做强校验（已有 ResultValidator 做长度等清洗）。
    """

    status: TaskStatus
    summary: str = ""
    files_changed: list[str] = field(default_factory=list)
    unresolved_issues: str = ""
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_structured(cls, result: StructuredResult) -> "AgentReport":
        return cls(
            status=result.status,
            summary=result.summary,
            files_changed=list(result.files_changed),
            unresolved_issues=result.unresolved_issues,
        )

    @classmethod
    def absent(cls, reason: str = "agent 未调用 submit_task_result") -> "AgentReport":
        """Agent 没调 submit 时的"缺失报告"占位"""
        return cls(
            status=TaskStatus.FAILED,
            summary=reason,
            files_changed=[],
            unresolved_issues=reason,
        )


@dataclass
class SystemObservation:
    """系统对 task 客观状态的观察。**事实**，task 的 final_status 据此而定。

    填充时机：
      - prepare_for_execution 失败 → workspace_final_state=NEVER_CREATED
      - agent ReAct 走完 + commit_workspace 失败 → CREATED_BUT_FAILED 或 CORRUPTED
      - DAG 收尾 merge 完成 → MERGED
      - DAG 收尾 merge 冲突 → COMMITTED_NOT_MERGED
    """

    workspace_final_state: WorkspaceFinalState
    """workspace 在 task 退出时的客观状态"""

    actual_files_landed: list[str] = field(default_factory=list)
    """**真正进入用户分支的文件清单**。空列表 = 没东西落地。
    与 agent_report.files_changed 可能不同 —— 后者是 agent 声称的，
    前者是系统从 git 实际看到的。"""

    failure_category: FailureCategory | None = None
    """失败分类（仅当系统判定为失败时填充）"""

    failure_reason: str = ""
    """人读得懂的失败说明（拼到错误日志和 master 提示词里）"""

    branch_name: str | None = None
    """task 的 subagent 分支（如果创建过）"""

    commit_hash: str | None = None
    """实际产生的 commit hash（如果 commit 成功）"""

    tool_call_counts: dict[str, int] = field(default_factory=dict)
    """该 task 内 subagent 调用每个工具的次数。
    用于 master 诊断行为模式 —— 例如 timeout 任务里 web_search×15 而 submit_task_result×0
    说明 agent 在死循环搜索而不是真的需要更多时间。"""

    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def succeeded(self) -> bool:
        """系统判定成功的唯一条件：产物**已**落到用户分支。

        两条路径都算落地：
          - MERGED：本任务的 worktree 直接 merge（leaf 路径）
          - MERGED_VIA_DESCENDANT：commit 经下游 leaf 间接进入主干（中间节点路径）

        committed-not-merged 不算成功 —— 它意味着 subagent 分支上有 commit
        但用户分支没有，对终端用户而言等同于"什么都没发生"。
        """
        return self.workspace_final_state in (
            WorkspaceFinalState.MERGED,
            WorkspaceFinalState.MERGED_VIA_DESCENDANT,
        )

    @property
    def is_terminal_failure(self) -> bool:
        """终态失败（已确定不再恢复）"""
        return self.workspace_final_state in (
            WorkspaceFinalState.NEVER_CREATED,
            WorkspaceFinalState.CREATED_BUT_FAILED,
            WorkspaceFinalState.CORRUPTED,
            WorkspaceFinalState.CLEANED_NO_PRODUCT,
        )


@dataclass
class TaskOutcome:
    """Task 的最终结果 —— 系统观察 + agent 自报合体。

    Master Agent 拿到的就是这个对象（经 TaskGraphResult 序列化后）。
    系统观察永远优先。
    """

    task_id: str
    agent_report: AgentReport
    system_observation: SystemObservation

    @property
    def final_status(self) -> TaskStatus:
        """**唯一**的真理来源。

        策略：
          - 系统观察 succeeded → 看 agent_report 是 SUCCESS / PARTIAL_SUCCESS
            （系统判落地了，但 agent 自己说部分成功，那就部分成功）
          - 系统观察 NOT_APPLICABLE → 走 agent_report（只读角色不需要落地）
          - 其他所有情况 → FAILED（不管 agent 说什么）

        关键：决不让 agent 的乐观自报覆盖系统观察的失败。
        """
        if self.system_observation.workspace_final_state == WorkspaceFinalState.NOT_APPLICABLE:
            return self.agent_report.status

        if self.system_observation.succeeded:
            # 系统判定落地了，agent 自报取个上限
            if self.agent_report.status == TaskStatus.PARTIAL_SUCCESS:
                return TaskStatus.PARTIAL_SUCCESS
            return TaskStatus.SUCCESS

        # 系统判定未落地 —— 不管 agent 说啥都是失败
        return TaskStatus.FAILED

    @property
    def discrepancy(self) -> str | None:
        """Agent 自报与系统观察的不一致描述。

        当 agent 报告"成功"但系统判定"失败"时，这是最重要的诚实信号 ——
        必须让 Master 看到这个差异，避免 Master 把 agent 的乐观当真理。

        返回 None = 没有不一致；返回字符串 = 给 Master 的人读说明。
        """
        agent_optimistic = self.agent_report.status == TaskStatus.SUCCESS
        system_failed = self.system_observation.is_terminal_failure
        if agent_optimistic and system_failed:
            cat = self.system_observation.failure_category
            cat_name = cat.value if cat else "unknown"
            reason = self.system_observation.failure_reason or "(无详细原因)"
            return (
                f"agent 自报 success，但系统观察该 task 实际失败 "
                f"[category={cat_name}]：{reason}。"
                f"agent 声称的 files_changed={self.agent_report.files_changed} "
                f"未真正进入用户分支。"
            )

        # 反向：agent 说失败但系统观察成功（极罕见）
        agent_pessimistic = self.agent_report.status == TaskStatus.FAILED
        if agent_pessimistic and self.system_observation.succeeded:
            return (
                "agent 自报 failed，但系统观察产物已落入用户分支。"
                "默认采纳系统观察。"
            )

        # 文件清单差异（即使 status 一致）
        if self.system_observation.succeeded:
            agent_files = set(self.agent_report.files_changed)
            actual_files = set(self.system_observation.actual_files_landed)
            extra = agent_files - actual_files
            missing = actual_files - agent_files
            if extra or missing:
                bits = []
                if extra:
                    bits.append(f"agent 声称改了但 git 没看到: {sorted(extra)[:5]}")
                if missing:
                    bits.append(f"git 实际有变更但 agent 没声称: {sorted(missing)[:5]}")
                return "；".join(bits)

        return None

    def to_master_context(self) -> str:
        """转成 Master Agent 可读的精简文本。

        优先表达事实（系统观察）+ 不一致警示（discrepancy）+ agent 摘要参考。
        """
        lines = [
            f"**任务**: `{self.task_id}`",
            f"**状态**: {self.final_status.value} (系统判定)",
        ]

        obs = self.system_observation
        if obs.workspace_final_state != WorkspaceFinalState.NOT_APPLICABLE:
            lines.append(f"**Worktree 终态**: {obs.workspace_final_state.value}")
            if obs.actual_files_landed:
                files_preview = ", ".join(obs.actual_files_landed[:5])
                if len(obs.actual_files_landed) > 5:
                    files_preview += f" ... 共 {len(obs.actual_files_landed)} 个"
                lines.append(f"**实际落入主干的文件**: {files_preview}")
            elif self.final_status == TaskStatus.SUCCESS:
                lines.append("**实际落入主干的文件**: (无文件变更)")

        if obs.failure_category is not None:
            lines.append(f"**失败分类**: {obs.failure_category.value}")
        if obs.failure_reason:
            lines.append(f"**失败原因**: {obs.failure_reason[:200]}")

        if self.agent_report.summary:
            lines.append(f"**Agent 摘要（仅参考）**: {self.agent_report.summary[:200]}")

        d = self.discrepancy
        if d:
            lines.append(f"**⚠️ 一致性差异**: {d}")

        return "\n".join(lines)


def build_outcome_for_no_worktree_task(
    task_id: str,
    agent_report: AgentReport,
) -> TaskOutcome:
    """便捷构造：不需要 worktree 的角色（Planner / Searcher）的 outcome。

    这种任务没有"产物落地"概念，agent 自报即事实。
    """
    return TaskOutcome(
        task_id=task_id,
        agent_report=agent_report,
        system_observation=SystemObservation(
            workspace_final_state=WorkspaceFinalState.NOT_APPLICABLE,
            actual_files_landed=[],
        ),
    )


def build_outcome_for_workspace_failure(
    task_id: str,
    failure_category: FailureCategory,
    failure_reason: str,
    branch_name: str | None = None,
) -> TaskOutcome:
    """便捷构造：worktree 创建/准备阶段就失败了，agent 还没机会跑。"""
    return TaskOutcome(
        task_id=task_id,
        agent_report=AgentReport.absent(
            reason=f"agent 未启动：{failure_reason}",
        ),
        system_observation=SystemObservation(
            workspace_final_state=WorkspaceFinalState.NEVER_CREATED,
            failure_category=failure_category,
            failure_reason=failure_reason,
            branch_name=branch_name,
        ),
    )


__all__ = [
    "WorkspaceFinalState",
    "AgentReport",
    "SystemObservation",
    "TaskOutcome",
    "build_outcome_for_no_worktree_task",
    "build_outcome_for_workspace_failure",
]
