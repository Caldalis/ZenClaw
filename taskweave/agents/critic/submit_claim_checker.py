"""
submit_task_result 自报核对器 — 防止 Subagent "假装完成"。

核对两条独立契约：
  C. Cleanup 删除申报：CleanupAgent 可用 delete_artifact 删除主仓库过程
     产物，但删除结果必须申报在 details.deleted_files；files_changed 仍
     必须为空，因为它只表示"进入 worktree commit/merge 闭环的改动"。
  B. 静态推断 vs 自报：角色没有写工具（write_file/edit_file/create_file）
     却在 files_changed 里声明改了文件 —— 这是物理不可能的（工具表里没有
     就调不到），属于"假装完成"模式。直接拒绝 submit 让 agent 重写。
  A. NOT_APPLICABLE 角色清空 files_changed：角色不进 worktree
     (requires_worktree=False) 时，即便它有写工具也无法走 commit/merge
     闭环，写出来的文件也不会进主干分支。这种角色的 files_changed 永远
     无法被系统验收，强制清空。
两条任一触发 → 拒绝；多次拒绝 → force-release（降级 status 并清字段）以
免 agent 撞 max_iterations。force-release 后 master 拿到 partial_success/
failed 信号，可以决定重派给合适的角色。
跟 ValidationAwareSubmitTool 的关系：
  这两个包装器各管一件事，叠加使用：
    FileClaimAwareSubmitTool   ←  外层（核对自报真实性）
      └─ ValidationAwareSubmitTool  ←  中层（验证闭环）
            └─ SubmitTaskResultTool  ←  内层（实际落账）
  没启用 validation 的角色（Planner/Searcher 等）：
    FileClaimAwareSubmitTool
      └─ SubmitTaskResultTool
"""

from __future__ import annotations

import json
from typing import Any

from taskweave.tools.base import Tool
from taskweave.utils.logging import get_logger

logger = get_logger(__name__)


# 写工具名称：拥有任一即视为"能落盘"
_WRITE_TOOL_NAMES = frozenset({
    "write_file", "edit_file", "create_file",
})
_DELETE_TOOL_NAMES = frozenset({
    "delete_artifact",
})

# 同名 file 在 force-release 第几次后强制放行（防 agent 死循环）
_FORCE_RELEASE_AFTER = 2


def _can_write_files(
    allowed_tools: list[str] | None,
    forbidden_tools: list[str] | None,
) -> bool:
    """从 agent 的工具白/黑名单推断是否能落盘文件。

    与 SubagentFactory._infer_requires_validation_for_dynamic_role 同口径都是
    看写工具有没有，但语义不同 —— 这里判 files_changed 的合法性，那里判
    validation 的默认值。两边各自演化更安全，所以不强行共享一个函数。
    """
    forbidden_set = {t.lower() for t in (forbidden_tools or [])}
    if _WRITE_TOOL_NAMES & forbidden_set:
        # 显式禁止写  一定不能落盘
        return False

    if allowed_tools:
        allowed_set = {t.lower() for t in allowed_tools}
        return bool(_WRITE_TOOL_NAMES & allowed_set)

    # allowed_tools 为空 表示 全部允许 能落盘
    return True


def _can_delete_artifacts(
    allowed_tools: list[str] | None,
    forbidden_tools: list[str] | None,
) -> bool:
    """从工具白/黑名单推断是否有受限删除过程产物的能力。"""
    forbidden_set = {t.lower() for t in (forbidden_tools or [])}
    if _DELETE_TOOL_NAMES & forbidden_set:
        return False

    if allowed_tools:
        allowed_set = {t.lower() for t in allowed_tools}
        return bool(_DELETE_TOOL_NAMES & allowed_set)

    return True


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


class FileClaimAwareSubmitTool(Tool):
    """submit_task_result 包装器：核对 files_changed 自报与角色实际能力的契约。

    构造时一次性吃下角色的工具白/黑名单和 worktree 标志，每次 submit 调用
    时按下面的表决定行为：

        | can_write | requires_worktree | files_changed 非空时           |
        |-----------|-------------------|--------------------------------|
        | False     | *                 | B 拒绝（教学/强制清+降级 failed）|
        | True      | False             | A 拒绝（教学/强制清+降级 partial）|
        | True      | True              | 透传（正常路径）                |

    files_changed 为空时永远透传（agent 自己说没改文件，没什么好核的）。
    """

    def __init__(
        self,
        inner_tool: Tool,
        allowed_tools: list[str] | None,
        forbidden_tools: list[str] | None,
        requires_worktree: bool,
        role_name: str,
    ):
        self._inner = inner_tool
        self._allowed = list(allowed_tools or [])
        self._forbidden = list(forbidden_tools or [])
        self._can_write = _can_write_files(allowed_tools, forbidden_tools)
        self._can_delete = _can_delete_artifacts(allowed_tools, forbidden_tools)
        self._requires_worktree = requires_worktree
        self._role_name = role_name
        self._rej_no_write_tools = 0
        self._rej_no_worktree = 0
        self._rej_cleanup_files_changed = 0

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def description(self) -> str:
        # 透传 inner 的 description（它已经被 ValidationAware 加工过）
        # 同时追加一段自报核对契约说明，让 agent 知道游戏规则
        base = self._inner.description
        addendum = (
            "\n\n**files_changed 自报核对**：\n"
            "- 你的角色工具表里如果**没有** `write_file` / `edit_file`，"
            "files_changed **必须**为空数组 `[]`，否则 submit 会被拒绝。\n"
            "- 你的角色如果不进 Git Worktree（`requires_worktree=false`），"
            "files_changed 也必须为空 —— 没有 commit/merge 闭环，写盘也没意义。\n"
            "- CleanupAgent 的删除结果是例外语义：删除必须走 `delete_artifact`，"
            "并把清单写进 `details.deleted_files`；`files_changed` 仍必须是 `[]`。\n"
            "- 这两条都是物理事实，不是建议。撒谎填 files_changed 不会提升任务"
            "完成度，反而会触发拒绝循环。\n\n"
            "**重要补充：合法的退出路径**\n"
            "- `submit_task_result(status='failed', files_changed=[], summary='...')` "
            "**永远是合法的**，本规则只拦『撒谎说改了文件』，不拦『诚实承认失败』。\n"
            "- 如果你尝试了一段时间但确实做不下去（比如外部 API 一直连不上、"
            "工具不支持你需要的操作、任务指令本身有矛盾等），**及时 submit failed 是正确选择**，"
            "不要硬撑到 max_iterations。Master 收到 failed 信号后会反思并重新派发，"
            "比你死循环更有效。\n"
            "- 同样，`submit_task_result(status='partial_success', files_changed=[<真改的文件>])` "
            "也合法 —— 老实说『我做了 X 但 Y 没做完』对系统是有用信号。"
        )
        return base + addendum

    @property
    def parameters(self) -> dict[str, Any]:
        return self._inner.parameters

    def to_tool_schema(self) -> dict[str, Any]:
        return self._inner.to_tool_schema()

    async def execute(self, **kwargs: Any) -> str:
        files_changed = _as_str_list(kwargs.get("files_changed"))
        deleted_files = self._extract_deleted_files(kwargs)

        if self._can_report_deleted_files():
            if files_changed:
                self._rej_cleanup_files_changed += 1
                if self._rej_cleanup_files_changed <= 1:
                    return self._build_cleanup_files_changed_rejection(
                        files_changed, deleted_files,
                    )
                return await self._force_release_cleanup_delete_claim(
                    kwargs, files_changed, deleted_files,
                )
        elif deleted_files:
            return self._build_invalid_deleted_files_rejection(deleted_files)

        if not files_changed:
            return await self._inner.execute(**kwargs)
        if not self._can_write:
            self._rej_no_write_tools += 1
            if self._rej_no_write_tools <= 1:
                return self._build_no_write_rejection(files_changed)
            return await self._force_release_no_write(kwargs, files_changed)
        if not self._requires_worktree:
            self._rej_no_worktree += 1
            if self._rej_no_worktree <= 1:
                return self._build_no_worktree_rejection(files_changed)
            return await self._force_release_no_worktree(kwargs, files_changed)
        return await self._inner.execute(**kwargs)

    def _can_report_deleted_files(self) -> bool:
        return (
            self._role_name.lower() in {"cleanupagent", "cleanup"}
            and self._can_delete
            and not self._requires_worktree
        )

    @staticmethod
    def _extract_deleted_files(kwargs: dict[str, Any]) -> list[str]:
        details = kwargs.get("details", {})
        if not isinstance(details, dict):
            return []
        return _as_str_list(details.get("deleted_files"))

    def _build_cleanup_files_changed_rejection(
        self,
        files_changed: list[str],
        deleted_files: list[str],
    ) -> str:
        delete_claims = deleted_files or files_changed
        delete_claims_json = json.dumps(delete_claims[:20], ensure_ascii=False)
        return (
            "**提交被阻止 (file_claim_check / cleanup_delete_claim)**\n\n"
            "CleanupAgent 可以删除过程产物，但删除不是 worktree commit/merge "
            "闭环里的 files_changed。`files_changed` 必须保持为空数组 `[]`。\n\n"
            f"你这次在 files_changed 里声明了：{files_changed[:5]}"
            f"{' ...' if len(files_changed) > 5 else ''}。\n\n"
            "请重新调用 submit_task_result，形状如下：\n"
            "`details.deleted_files` 记录删除清单，`files_changed` 保持空：\n"
            "```json\n"
            "{\n"
            '  "status": "success",\n'
            '  "files_changed": [],\n'
            f'  "details": {{"deleted_files": {delete_claims_json}}}\n'
            "}\n"
            "```\n"
            "summary 里可以继续说明哪些文件已删除、哪些本来不存在。"
        )

    async def _force_release_cleanup_delete_claim(
        self,
        kwargs: dict[str, Any],
        files_changed: list[str],
        deleted_files: list[str],
    ) -> str:
        logger.warning(
            "FileClaimAwareSubmitTool: CleanupAgent 反复把删除结果写进 "
            "files_changed=%s，强制清空并迁移到 details.deleted_files",
            files_changed,
        )
        patched = dict(kwargs)
        patched["files_changed"] = []

        details = patched.get("details", {})
        if not isinstance(details, dict):
            details = {}
        else:
            details = dict(details)
        if not deleted_files:
            details["deleted_files"] = list(files_changed)
        patched["details"] = details

        existing = (kwargs.get("unresolved_issues") or "").strip()
        note = (
            "[系统注记] CleanupAgent 的删除结果不属于 files_changed；"
            "files_changed 已强制清空，删除清单请看 details.deleted_files。"
        )
        patched["unresolved_issues"] = (
            f"{existing} | {note}" if existing else note
        )
        return await self._inner.execute(**patched)

    def _build_invalid_deleted_files_rejection(
        self,
        deleted_files: list[str],
    ) -> str:
        return (
            "**提交被阻止 (file_claim_check / invalid_deleted_files_claim)**\n\n"
            "`details.deleted_files` 只允许 CleanupAgent 通过 `delete_artifact` "
            "申报主仓库过程产物删除结果。当前角色不能申报 deleted_files："
            f"{deleted_files[:5]}{' ...' if len(deleted_files) > 5 else ''}。\n\n"
            "请移除 `details.deleted_files` 后重新提交；如果确实需要清理过程"
            "产物，应由 Master 单独派 CleanupAgent 执行 cleanup-only DAG。"
        )

    def _build_no_write_rejection(self, files_changed: list[str]) -> str:
        return (
            "**提交被阻止 (file_claim_check / no_write_tools)**\n\n"
            f"你的角色 `{self._role_name}` 工具表里**没有** `write_file` / "
            "`edit_file` / `create_file` 等写工具，物理上不可能创建或修改任何"
            f"文件，但你的 files_changed 字段声明改了：{files_changed[:5]}"
            f"{' ...' if len(files_changed) > 5 else ''}。\n\n"
            "请按以下方式之一处理：\n"
            "1. 如果你的产出是**文字方案 / 分析 / 规划**（不需要落盘），"
            "把 `files_changed` 设为空数组 `[]`，完整方案直接写在 `summary` "
            "或 `details` 字段里返回。\n"
            "2. 如果任务**确实需要落盘文件**，这是**任务派发错误** —— 把 "
            "`status` 设为 `failed`，在 `unresolved_issues` 里写："
            "「当前角色没有 write_file 权限，请重新派给 CoderAgent 或带 "
            "write_file 的自定义角色」。Master 会重新派发。\n\n"
            "请重新调用 submit_task_result。"
        )

    async def _force_release_no_write(
        self,
        kwargs: dict[str, Any],
        files_changed: list[str],
    ) -> str:
        logger.warning(
            "FileClaimAwareSubmitTool: 角色 %s 没有写工具但反复声称改了文件 %s "
            "（拒绝 %d 次），强制清空 files_changed 并降级 status=failed",
            self._role_name, files_changed, self._rej_no_write_tools,
        )
        patched = dict(kwargs)
        patched["files_changed"] = []
        patched["status"] = "failed"
        existing = (kwargs.get("unresolved_issues") or "").strip()
        note = (
            f"[系统注记] 角色 {self._role_name} 没有 write 工具但反复声称改了"
            f"文件 {files_changed[:5]}；files_changed 已被强制清空，status 降级"
            "为 failed。这通常意味着 master 把落盘任务派错了角色，"
            "建议改派给 CoderAgent 或带 write_file 的自定义角色。"
        )
        patched["unresolved_issues"] = (
            f"{existing} | {note}" if existing else note
        )
        return await self._inner.execute(**patched)

    def _build_no_worktree_rejection(self, files_changed: list[str]) -> str:
        return (
            "**提交被阻止 (file_claim_check / no_worktree)**\n\n"
            f"你的角色 `{self._role_name}` 不进 Git Worktree "
            "(`requires_worktree=false`)，意味着你即便能写文件，写出来的"
            "东西也不会经过 commit/merge 闭环进入主干分支 —— 系统无法验收，"
            "下游任务也看不到。\n"
            f"但你的 files_changed 字段声明改了：{files_changed[:5]}"
            f"{' ...' if len(files_changed) > 5 else ''}。\n\n"
            "请重新提交：\n"
            "- 如果你只是输出文字内容（设计方案 / 分析报告），"
            "把 `files_changed` 设为 `[]`，内容写到 `summary`。\n"
            "- 如果任务确实需要落盘，**这是任务派发错误** —— 把 status 设为 "
            "`failed`，在 `unresolved_issues` 里说明「当前角色 "
            "requires_worktree=false，无法落盘文件，请改派给 worktree 角色」。"
        )

    async def _force_release_no_worktree(
        self,
        kwargs: dict[str, Any],
        files_changed: list[str],
    ) -> str:
        logger.warning(
            "FileClaimAwareSubmitTool: 角色 %s 不进 worktree 但反复声称改了"
            "文件 %s （拒绝 %d 次），强制清空 files_changed 并降级 "
            "status=partial_success",
            self._role_name, files_changed, self._rej_no_worktree,
        )
        patched = dict(kwargs)
        patched["files_changed"] = []
        if (patched.get("status") or "success") == "success":
            patched["status"] = "partial_success"
        existing = (kwargs.get("unresolved_issues") or "").strip()
        note = (
            f"[系统注记] 角色 {self._role_name} 不进 worktree 但反复声称改了"
            f"文件 {files_changed[:5]}；files_changed 已被强制清空，status 降级"
            "为 partial_success。任务可能有文字产出但没有真实落盘，"
            "若需要文件请改派给 worktree 角色。"
        )
        patched["unresolved_issues"] = (
            f"{existing} | {note}" if existing else note
        )
        return await self._inner.execute(**patched)


__all__ = [
    "FileClaimAwareSubmitTool",
]
