#!/usr/bin/env python3
# Harness: all mechanisms combined -- the complete cockpit for the model.
"""
s_full.py - Full Reference Agent

Capstone implementation combining every mechanism from s01-s11.
Session s12 (task-aware worktree isolation) is taught separately.
NOT a teaching session -- this is the "put it all together" reference.

    +------------------------------------------------------------------+
    |                        FULL AGENT                                 |
    |                                                                   |
    |  System prompt (s05 skills, task-first + optional todo nag)      |
    |                                                                   |
    |  Before each LLM call:                                            |
    |  +--------------------+  +------------------+  +--------------+  |
    |  | Microcompact (s06) |  | Drain bg (s08)   |  | Check inbox  |  |
    |  | Auto-compact (s06) |  | notifications    |  | (s09)        |  |
    |  +--------------------+  +------------------+  +--------------+  |
    |                                                                   |
    |  Tool dispatch (s02 pattern):                                     |
    |  +--------+----------+----------+---------+-----------+          |
    |  | bash   | read     | write    | edit    | TodoWrite |          |
    |  | task   | load_sk  | compress | bg_run  | bg_check  |          |
    |  | t_crt  | t_get    | t_upd    | t_list  | spawn_tm  |          |
    |  | list_tm| send_msg | rd_inbox | bcast   | shutdown  |          |
    |  | plan   | idle     | claim    |         |           |          |
    |  +--------+----------+----------+---------+-----------+          |
    |                                                                   |
    |  Subagent (s04):  spawn -> work -> return summary                 |
    |  Teammate (s09):  spawn -> work -> idle -> auto-claim (s11)      |
    |  Shutdown (s10):  request_id handshake                            |
    |  Plan gate (s10): submit -> approve/reject                        |
    +------------------------------------------------------------------+

    REPL commands: /compact /tasks /team /inbox
"""

import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from queue import Queue

from anthropic import Anthropic
from dotenv import load_dotenv

# ============================================================
# SECTION: 初始化
# ============================================================

# 加载环境变量
load_dotenv(override=True)
# 若设置了自定义 BASE_URL，则移除默认的 AUTH_TOKEN（避免冲突）
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()    # 工作目录：所有文件操作都限制在此目录下（沙盒安全）
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))    # 创建客户端
MODEL = os.environ["MODEL_ID"]  # 创建模型

# 各功能目录
TEAM_DIR       = WORKDIR / ".team"        # 团队配置（成员列表、状态）
INBOX_DIR      = TEAM_DIR / "inbox"       # 消息收件箱（每人一个 .jsonl 文件）
TASKS_DIR      = WORKDIR / ".tasks"       # 持久任务（每个任务一个 JSON 文件）
SKILLS_DIR     = WORKDIR / "skills"       # 技能文件夹（SKILL.md）
TRANSCRIPT_DIR = WORKDIR / ".transcripts" # 压缩前的对话存档

# 参数配置
TOKEN_THRESHOLD = 100000   # 触发自动压缩的 token 估算阈值
POLL_INTERVAL   = 5        # Teammate 空闲轮询间隔（秒）
IDLE_TIMEOUT    = 60       # Teammate 无任务超时后自动关闭（秒）

# 合法的消息类型（发送到收件箱时的 type 字段）
VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request",
                   "shutdown_response", "plan_approval_response"}


# ============================================================
# SECTION: base_tools (s02) — 基础工具函数
# ============================================================

def safe_path(p: str) -> Path:
    """
    将相对路径解析为绝对路径，并确保不逃出 WORKDIR（路径遍历防护）。
    任何试图访问 /etc/passwd 或 ../../secret 的路径都会被拒绝。
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    """
    在 WORKDIR 下执行 shell 命令。
    - 内置黑名单：阻止 rm -rf /、sudo 等危险命令
    - 超时 120 秒
    - stdout + stderr 合并，截断到 50000 字符
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    """
    读取文件内容（纯文本）。
    limit 参数限制行数，超出时追加"... (N more)"提示。
    """
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    """
    写入文件（若目录不存在则自动创建）。
    返回写入字节数，方便 LLM 确认。
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    精确替换文件中的一段文本（只替换第一次出现）。
    若找不到 old_text 则报错，防止静默修改错误位置。
    """
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# SECTION: todos (s03) — 短期清单管理
# ============================================================

class TodoManager:
    """
    管理一个内存中的 Todo 列表（最多 20 条）。

    【设计意图】
    用于追踪当前任务中的子步骤，帮助 LLM 保持进度意识。
    不持久化到磁盘（会话结束即清空）。对于需要跨会话的任务，
    使用 TaskManager（file_tasks）。

    状态机：pending → in_progress → completed
    约束：同时只允许一个 in_progress 项目，防止 LLM 分心。
    """
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        """
        全量替换 todo 列表。LLM 每次都传入完整列表。
        验证规则：
          - content 非空
          - status 必须是 pending/in_progress/completed
          - activeForm 非空（显示当前正在做的具体动作）
          - 最多一个 in_progress
        返回渲染后的文本，LLM 可直接阅读。
        """
        validated, ip = [], 0
        for i, item in enumerate(items):
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()
            af = str(item.get("activeForm", "")).strip()
            if not content: raise ValueError(f"Item {i}: content required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {i}: invalid status '{status}'")
            if not af: raise ValueError(f"Item {i}: activeForm required")
            if status == "in_progress": ip += 1
            validated.append({"content": content, "status": status, "activeForm": af})
        if len(validated) > 20: raise ValueError("Max 20 todos")
        if ip > 1: raise ValueError("Only one in_progress allowed")
        self.items = validated
        return self.render()

    def render(self) -> str:
        """将 todo 列表渲染为人类可读的文本格式。"""
        if not self.items: return "No todos."
        lines = []
        for item in self.items:
            m = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}.get(item["status"], "[?]")
            suffix = f" <- {item['activeForm']}" if item["status"] == "in_progress" else ""
            lines.append(f"{m} {item['content']}{suffix}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

    def has_open_items(self) -> bool:
        return any(item.get("status") != "completed" for item in self.items)


# ============================================================
# SECTION: subagent (s04) — 子智能体
# ============================================================

def run_subagent(prompt: str, agent_type: str = "Explore") -> str:
    """
    在当前进程内同步运行一个"一次性"子智能体。

    【与 Teammate 的区别】
    - subagent：同步阻塞、一次性、有固定迭代上限（30轮）
    - teammate：异步线程、持久存在、可以空闲等待并自动认领任务

    agent_type:
      "Explore"         — 只读工具（bash + read_file）
      "general-purpose" — 读写工具（bash + read/write/edit）

    返回子智能体最后一条文本回复作为"汇报摘要"。
    """
    sub_tools = [
        {"name": "bash", "description": "Run command.",
         "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        {"name": "read_file", "description": "Read file.",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    ]
    # general-purpose 子智能体额外拥有写入和编辑权限
    if agent_type != "Explore":
        sub_tools += [
            {"name": "write_file", "description": "Write file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Edit file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
        ]
    sub_handlers = {
        "bash": lambda **kw: run_bash(kw["command"]),
        "read_file": lambda **kw: run_read(kw["path"]),
        "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
        "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    }
    sub_msgs = [{"role": "user", "content": prompt}]
    resp = None
    # 最多 30 轮工具调用，防止无限循环
    for _ in range(30):
        resp = client.messages.create(model=MODEL, messages=sub_msgs, tools=sub_tools, max_tokens=8000)
        sub_msgs.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break   # 不再调用工具，子智能体完成
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                h = sub_handlers.get(b.name, lambda **kw: "Unknown tool")
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(h(**b.input))[:50000]})
        sub_msgs.append({"role": "user", "content": results})
    # 提取最后一条文本响应作为摘要返回给主智能体
    if resp:
        return "".join(b.text for b in resp.content if hasattr(b, "text")) or "(no summary)"
    return "(subagent failed)"


# ============================================================
# SECTION: skills (s05) — 技能加载器
# ============================================================

class SkillLoader:
    """
       从 skills/ 目录递归加载所有 SKILL.md 文件。

       【SKILL.md 格式】
       ---
       name: git
       description: Git workflow and branching strategy
       ---
       （技能正文 Markdown 内容）

       【工作流程】
       LLM 首先通过 skills 描述列表了解有哪些技能可用（注入到 system prompt）。
       需要时调用 load_skill(name)，将完整技能内容注入到对话上下文。
       这种"按需加载"模式节省了 token，避免一次性塞入所有知识。
       """
    def __init__(self, skills_dir: Path):
        self.skills = {}
        if skills_dir.exists():
            for f in sorted(skills_dir.rglob("SKILL.md")):
                text = f.read_text()
                # 解析 YAML front-matter（--- ... ---）
                match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
                meta, body = {}, text
                if match:
                    for line in match.group(1).strip().splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            meta[k.strip()] = v.strip()
                    body = match.group(2).strip()
                name = meta.get("name", f.parent.name)
                self.skills[name] = {"meta": meta, "body": body}

    def descriptions(self) -> str:
        """返回所有技能的名称+描述，用于 system prompt 中的技能目录。"""
        if not self.skills: return "(no skills)"
        return "\n".join(f"  - {n}: {s['meta'].get('description', '-')}" for n, s in self.skills.items())

    def load(self, name: str) -> str:
        """
        按名称加载完整技能内容，用 <skill> 标签包裹返回。
        LLM 会将其作为权威参考来执行对应任务。
        """
        s = self.skills.get(name)
        if not s: return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{s['body']}\n</skill>"


# ============================================================
# SECTION: compression (s06) — 上下文压缩
# ============================================================

def estimate_tokens(messages: list) -> int:
    """
    快速估算消息列表的 token 数（用 JSON 字节数 / 4 近似）。
    不精确但足够触发压缩决策，避免每次都调用 tokenizer API。
    """
    return len(json.dumps(messages, default=str)) // 4

def microcompact(messages: list):
    """
    微压缩：将旧的工具调用结果（tool_result）内容清空为 "[cleared]"。

    【策略】保留最近 3 条工具结果，清除更早的。
    这样 LLM 仍能看到最近的工具输出，同时节省大量 token。
    无损：清除的是已经被 LLM 处理过的旧结果，不影响推理。
    """
    indices = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    indices.append(part)
    if len(indices) <= 3:
        return
    # 只清除旧的（保留最后 3 个）
    for part in indices[:-3]:
        if isinstance(part.get("content"), str) and len(part["content"]) > 100:
            part["content"] = "[cleared]"

def auto_compact(messages: list) -> list:
    """
       自动压缩（有损）：当 token 超过阈值时触发。

       【步骤】
       1. 将完整对话存档到 .transcripts/ 目录（防止数据丢失）
       2. 取对话末尾 80000 字符交给 LLM 生成摘要
       3. 返回只含摘要的新消息列表

       摘要后的对话从"接力点"继续，LLM 不会丢失关键上下文。
       """
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    # 取末尾部分生成摘要（最重要的信息通常在最后）
    conv_text = json.dumps(messages, default=str)[-80000:]
    resp = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"Summarize for continuity:\n{conv_text}"}],
        max_tokens=2000,
    )
    summary = resp.content[0].text
    # 返回压缩后的对话：只有一条消息，包含摘要和存档路径
    return [
        {"role": "user", "content": f"[Compressed. Transcript: {path}]\n{summary}"},
    ]


# ============================================================
# SECTION: file_tasks (s07) — 持久任务管理
# ============================================================

class TaskManager:
    """
       基于文件的持久任务看板（每个任务存为 .tasks/task_N.json）。

       【与 TodoManager 的区别】
       - TodoManager：内存、短期、当前会话的子步骤清单
       - TaskManager：文件持久、跨会话、支持多智能体协作（认领/阻塞）

       状态：pending → in_progress → completed | deleted
       支持任务间依赖（blockedBy）：只有依赖的任务完成后，当前任务才可被认领。
       """
    def __init__(self):
        TASKS_DIR.mkdir(exist_ok=True)

    def _next_id(self) -> int:
        """自增 ID：扫描现有任务文件，取最大 ID + 1。"""
        ids = [int(f.stem.split("_")[1]) for f in TASKS_DIR.glob("task_*.json")]
        return max(ids, default=0) + 1

    def _load(self, tid: int) -> dict:
        p = TASKS_DIR / f"task_{tid}.json"
        if not p.exists(): raise ValueError(f"Task {tid} not found")
        return json.loads(p.read_text())

    def _save(self, task: dict):
        (TASKS_DIR / f"task_{task['id']}.json").write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        """创建新任务，初始状态 pending，无 owner，无依赖。"""
        task = {"id": self._next_id(), "subject": subject, "description": description,
                "status": "pending", "owner": None, "blockedBy": []}
        self._save(task)
        return json.dumps(task, indent=2)

    def get(self, tid: int) -> str:
        return json.dumps(self._load(tid), indent=2)

    def update(self, tid: int, status: str = None,
               add_blocked_by: list = None, remove_blocked_by: list = None) -> str:
        """
        更新任务状态或依赖关系。
        完成时（completed）自动从其他任务的 blockedBy 列表中移除此 ID。
        删除时（deleted）直接从磁盘移除文件。
        """
        task = self._load(tid)
        if status:
            task["status"] = status
            if status == "completed":
                # 级联解锁：遍历所有任务，移除对此任务的依赖
                for f in TASKS_DIR.glob("task_*.json"):
                    t = json.loads(f.read_text())
                    if tid in t.get("blockedBy", []):
                        t["blockedBy"].remove(tid)
                        self._save(t)
            if status == "deleted":
                (TASKS_DIR / f"task_{tid}.json").unlink(missing_ok=True)
                return f"Task {tid} deleted"
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if remove_blocked_by:
            task["blockedBy"] = [x for x in task["blockedBy"] if x not in remove_blocked_by]
        self._save(task)
        return json.dumps(task, indent=2)

    def list_all(self) -> str:
        """列出所有任务，格式：[状态] #ID: 主题 @owner (blocked by: [...])"""
        tasks = [json.loads(f.read_text()) for f in sorted(TASKS_DIR.glob("task_*.json"))]
        if not tasks: return "No tasks."
        lines = []
        for t in tasks:
            m = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            owner = f" @{t['owner']}" if t.get("owner") else ""
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{m} #{t['id']}: {t['subject']}{owner}{blocked}")
        return "\n".join(lines)

    def claim(self, tid: int, owner: str) -> str:
        """认领任务：设置 owner 并将状态改为 in_progress。"""
        task = self._load(tid)
        task["owner"] = owner
        task["status"] = "in_progress"
        self._save(task)
        return f"Claimed task #{tid} for {owner}"


# ============================================================
# SECTION: background (s08) — 后台任务管理
# ============================================================

class BackgroundManager:
    """
    在独立线程中运行耗时 shell 命令，不阻塞主智能体循环。

    【工作流程】
    1. background_run(cmd) → 立即返回任务 ID（8位 UUID）
    2. 主循环每轮开始时调用 drain() 收集完成通知
    3. 通知作为 <background-results> 注入到下一次 LLM 调用前

    适用场景：测试套件、编译、代码检查等可能运行数十秒的命令。
    """
    def __init__(self):
        self.tasks = {}
        self.notifications = Queue()

    def run(self, command: str, timeout: int = 120) -> str:
        """启动后台任务，返回任务 ID。"""
        tid = str(uuid.uuid4())[:8]
        self.tasks[tid] = {"status": "running", "command": command, "result": None}
        threading.Thread(target=self._exec, args=(tid, command, timeout), daemon=True).start()
        return f"Background task {tid} started: {command[:80]}"

    def _exec(self, tid: str, command: str, timeout: int):
        """实际执行线程：运行命令，完成后把结果推入通知队列。"""
        try:
            r = subprocess.run(command, shell=True, cwd=WORKDIR,
                               capture_output=True, text=True, timeout=timeout)
            output = (r.stdout + r.stderr).strip()[:50000]
            self.tasks[tid].update({"status": "completed", "result": output or "(no output)"})
        except Exception as e:
            self.tasks[tid].update({"status": "error", "result": str(e)})
        # 只放摘要（前 500 字符）进通知队列，完整结果在 tasks[tid] 中
        self.notifications.put({"task_id": tid, "status": self.tasks[tid]["status"],
                                "result": self.tasks[tid]["result"][:500]})

    def check(self, tid: str = None) -> str:
        """查询指定任务状态，或列出所有后台任务。"""
        if tid:
            t = self.tasks.get(tid)
            return f"[{t['status']}] {t.get('result') or '(running)'}" if t else f"Unknown: {tid}"
        return "\n".join(f"{k}: [{v['status']}] {v['command'][:60]}" for k, v in self.tasks.items()) or "No bg tasks."

    def drain(self) -> list:
        """取出并返回所有待消费的完成通知（清空队列）。"""
        notifs = []
        while not self.notifications.empty():
            notifs.append(self.notifications.get_nowait())
        return notifs


# ============================================================
# SECTION: messaging (s09) — 消息总线
# ============================================================

class MessageBus:
    """
    基于文件的进程内消息系统（同一进程的多个线程共享文件系统）。

    【存储格式】
    .team/inbox/<接收者名称>.jsonl — 每行一条 JSON 消息
    读取时清空文件（消费一次性语义）。

    【消息类型】
    - message             : 普通点对点消息
    - broadcast           : 群发消息
    - shutdown_request    : 请求 teammate 关闭
    - shutdown_response   : teammate 确认关闭
    - plan_approval_response : 主智能体对方案的审批结果
    """
    def __init__(self):
        INBOX_DIR.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        """向指定收件人追加一条消息（append-only，不覆盖）。"""
        msg = {"type": msg_type, "from": sender, "content": content,
               "timestamp": time.time()}
        if extra: msg.update(extra)
        with open(INBOX_DIR / f"{to}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        """
        读取并清空指定收件人的收件箱。
        返回消息列表（空则返回 []）。
        """
        path = INBOX_DIR / f"{name}.jsonl"
        if not path.exists(): return []
        msgs = [json.loads(l) for l in path.read_text().strip().splitlines() if l]
        path.write_text("")
        return msgs

    def broadcast(self, sender: str, content: str, names: list) -> str:
        """向所有成员（除发送者自己）发送广播消息。"""
        count = 0
        for n in names:
            if n != sender:
                self.send(sender, n, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


# ============================================================
# SECTION: shutdown + plan tracking (s10) — 关闭协议 & 计划审批
# ============================================================

shutdown_requests = {}  # 待处理的关闭请求：{request_id -> {target, status}}
plan_requests = {}  # 待处理的计划审批请求：{request_id -> {from, status}}


# ============================================================
# SECTION: team (s09/s11) — 团队成员管理
# ============================================================
class TeammateManager:
    """
       管理持久性团队成员（每人一个独立线程）。

       【Teammate 生命周期】
       spawn(name, role, prompt)
         │
         ▼
       工作阶段（WORK PHASE）
         ├─ 每轮：读收件箱 → LLM 调用 → 执行工具
         ├─ 收到 shutdown_request → 退出
         └─ 调用 idle 工具 → 进入空闲阶段
         │
         ▼
       空闲阶段（IDLE PHASE）—— 每 POLL_INTERVAL 秒检查：
         ├─ 收到消息 → 返回工作阶段
         ├─ 有未认领且未阻塞的任务 → 自动认领 → 返回工作阶段（s11）
         └─ 超时 IDLE_TIMEOUT 秒无事 → 自动关闭

       【配置持久化】
       .team/config.json 记录成员列表和状态，进程重启后可恢复。
       """
    def __init__(self, bus: MessageBus, task_mgr: TaskManager):
        TEAM_DIR.mkdir(exist_ok=True)
        self.bus = bus
        self.task_mgr = task_mgr
        self.config_path = TEAM_DIR / "config.json"
        self.config = self._load()
        self.threads = {}   # name -> Thread（目前未使用，为未来扩展预留）

    def _load(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find(self, name: str) -> dict:
        """按名称查找成员字典。"""
        for m in self.config["members"]:
            if m["name"] == name: return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """
        创建或重启 teammate。
        - 若成员已存在且处于 idle/shutdown，允许重新激活
        - 否则创建新成员
        - 启动独立守护线程运行 _loop
        """
        member = self._find(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save()
        threading.Thread(target=self._loop, args=(name, role, prompt), daemon=True).start()
        return f"Spawned '{name}' (role: {role})"

    def _set_status(self, name: str, status: str):
        """线程安全地更新成员状态并持久化。"""
        member = self._find(name)
        if member:
            member["status"] = status
            self._save()

    def _loop(self, name: str, role: str, prompt: str):
        """
        Teammate 主循环（在独立线程中运行）。

        【工作阶段】最多 50 轮 LLM 调用：
          - 每轮先读收件箱（处理消息/关闭请求）
          - LLM 调用 + 工具执行
          - 调用 idle 工具时跳出，进入空闲阶段

        【空闲阶段】每 POLL_INTERVAL 秒：
          - 检查收件箱 → 有消息则恢复工作
          - 检查任务看板 → 有未认领任务则自动认领并恢复工作（s11 自动认领）
          - 超时 → 标记为 shutdown 并退出线程
        """
        team_name = self.config["team_name"]
        sys_prompt = (f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
                      f"Use idle when done with current work. You may auto-claim tasks.")
        messages = [{"role": "user", "content": prompt}]
        # Teammate 可用的工具（比主智能体简化）
        tools = [
            {"name": "bash", "description": "Run command.", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"name": "read_file", "description": "Read file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "write_file", "description": "Write file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Edit file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
            {"name": "send_message", "description": "Send message.", "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}}, "required": ["to", "content"]}},
            {"name": "idle", "description": "Signal no more work.", "input_schema": {"type": "object", "properties": {}}},
            {"name": "claim_task", "description": "Claim task by ID.", "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
        ]
        while True:
            # ──────────────────────── 工作阶段 ────────────────────────
            for _ in range(50):
                # 先检查收件箱（可能有关闭请求）
                inbox = self.bus.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return  # 立即退出线程
                    # 将其他消息追加到对话历史，让 LLM 感知
                    messages.append({"role": "user", "content": json.dumps(msg)})
                try:
                    response = client.messages.create(
                        model=MODEL, system=sys_prompt, messages=messages,
                        tools=tools, max_tokens=8000)
                except Exception:
                    self._set_status(name, "shutdown")
                    return  # API 错误时优雅退出
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    break   # LLM 决定停止（结束工作阶段）
                # 执行工具
                results = []
                idle_requested = False
                for block in response.content:
                    if block.type == "tool_use":
                        if block.name == "idle":
                            idle_requested = True
                            output = "Entering idle phase."
                        elif block.name == "claim_task":
                            output = self.task_mgr.claim(block.input["task_id"], name)
                        elif block.name == "send_message":
                            output = self.bus.send(name, block.input["to"], block.input["content"])
                        else:
                            dispatch = {"bash": lambda **kw: run_bash(kw["command"]),
                                        "read_file": lambda **kw: run_read(kw["path"]),
                                        "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
                                        "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"])}
                            output = dispatch.get(block.name, lambda **kw: "Unknown")(**block.input)
                        print(f"  [{name}] {block.name}: {str(output)[:120]}")
                        results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                messages.append({"role": "user", "content": results})
                if idle_requested:
                    break   # 跳出工作循环，进入空闲阶段

            # ──────────────────────── 空闲阶段 ────────────────────────
            self._set_status(name, "idle")
            resume = False
            for _ in range(IDLE_TIMEOUT // max(POLL_INTERVAL, 1)):
                time.sleep(POLL_INTERVAL)
                # 检查收件箱
                inbox = self.bus.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break
                # s11：自动认领未分配且未阻塞的任务
                unclaimed = []
                for f in sorted(TASKS_DIR.glob("task_*.json")):
                    t = json.loads(f.read_text())
                    if t.get("status") == "pending" and not t.get("owner") and not t.get("blockedBy"):
                        unclaimed.append(t)
                if unclaimed:
                    task = unclaimed[0]
                    self.task_mgr.claim(task["id"], name)
                    # 身份重注入：若对话被压缩过（消息很短），重新插入身份信息防止 LLM 忘记自己是谁
                    if len(messages) <= 3:
                        messages.insert(0, {"role": "user", "content":
                            f"<identity>You are '{name}', role: {role}, team: {team_name}.</identity>"})
                        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})
                    messages.append({"role": "user", "content":
                        f"<auto-claimed>Task #{task['id']}: {task['subject']}\n{task.get('description', '')}</auto-claimed>"})
                    messages.append({"role": "assistant", "content": f"Claimed task #{task['id']}. Working on it."})
                    resume = True
                    break
            if not resume:
                # 超时无事可做，自动关闭
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")

    def list_all(self) -> str:
        """显示团队成员列表及其当前状态。"""
        if not self.config["members"]: return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


# ============================================================
# SECTION: global_instances — 全局单例
# ============================================================

# 所有管理器在模块加载时初始化，整个进程共享同一实例
TODO = TodoManager()
SKILLS = SkillLoader(SKILLS_DIR)
TASK_MGR = TaskManager()
BG = BackgroundManager()
BUS = MessageBus()
TEAM = TeammateManager(BUS, TASK_MGR)


# ============================================================
# SECTION: system_prompt — LLM 系统提示词
# ============================================================

SYSTEM = f"""You are a coding agent at {WORKDIR}. Use tools to solve tasks.
Prefer task_create/task_update/task_list for multi-step work. Use TodoWrite for short checklists.
Use task for subagent delegation. Use load_skill for specialized knowledge.
Skills: {SKILLS.descriptions()}"""


# ============================================================
# SECTION: shutdown_protocol (s10) — 关闭协议
# ============================================================

def handle_shutdown_request(teammate: str) -> str:
    """
    向指定 teammate 发送关闭请求（握手协议）。
    生成唯一 request_id 用于追踪确认响应。
    Teammate 在下一轮收件箱读取时会看到此消息并退出线程。
    """
    req_id = str(uuid.uuid4())[:8]
    shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send("lead", teammate, "Please shut down.", "shutdown_request", {"request_id": req_id})
    return f"Shutdown request {req_id} sent to '{teammate}'"


# ============================================================
# SECTION: plan_approval (s10) — 计划审批
# ============================================================

def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    """
    审批或拒绝 teammate 提交的执行计划。
    结果通过消息总线回传给 teammate，teammate 据此决定是否继续执行。
    """
    req = plan_requests.get(request_id)
    if not req: return f"Error: Unknown plan request_id '{request_id}'"
    req["status"] = "approved" if approve else "rejected"
    BUS.send("lead", req["from"], feedback, "plan_approval_response",
             {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"Plan {req['status']} for '{req['from']}'"


# ============================================================
# SECTION: tool_dispatch (s02) — 工具注册表
# ============================================================

TOOL_HANDLERS = {
    "bash":             lambda **kw: run_bash(kw["command"]),
    "read_file":        lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":       lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":        lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "TodoWrite":        lambda **kw: TODO.update(kw["items"]),
    "task":             lambda **kw: run_subagent(kw["prompt"], kw.get("agent_type", "Explore")),
    "load_skill":       lambda **kw: SKILLS.load(kw["name"]),
    "compress":         lambda **kw: "Compressing...",
    "background_run":   lambda **kw: BG.run(kw["command"], kw.get("timeout", 120)),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
    "task_create":      lambda **kw: TASK_MGR.create(kw["subject"], kw.get("description", "")),
    "task_get":         lambda **kw: TASK_MGR.get(kw["task_id"]),
    "task_update":      lambda **kw: TASK_MGR.update(kw["task_id"], kw.get("status"), kw.get("add_blocked_by"), kw.get("remove_blocked_by")),
    "task_list":        lambda **kw: TASK_MGR.list_all(),
    "spawn_teammate":   lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":   lambda **kw: TEAM.list_all(),
    "send_message":     lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":       lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":        lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request": lambda **kw: handle_shutdown_request(kw["teammate"]),
    "plan_approval":    lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    "idle":             lambda **kw: "Lead does not idle.",
    "claim_task":       lambda **kw: TASK_MGR.claim(kw["task_id"], "lead"),
}

# TOOLS：传给 LLM 的工具 Schema（JSON Schema 格式）
# LLM 根据这些 schema 决定调用哪个工具、传什么参数
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "TodoWrite", "description": "Update task tracking list.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "activeForm": {"type": "string"}}, "required": ["content", "status", "activeForm"]}}}, "required": ["items"]}},
    {"name": "task", "description": "Spawn a subagent for isolated exploration or work.",
     "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}, "agent_type": {"type": "string", "enum": ["Explore", "general-purpose"]}}, "required": ["prompt"]}},
    {"name": "load_skill", "description": "Load specialized knowledge by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "compress", "description": "Manually compress conversation context.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "background_run", "description": "Run command in background thread.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}},
    {"name": "check_background", "description": "Check background task status.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}}},
    {"name": "task_create", "description": "Create a persistent file task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_get", "description": "Get task details by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    {"name": "task_update", "description": "Update task status or dependencies.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "deleted"]}, "add_blocked_by": {"type": "array", "items": {"type": "integer"}}, "remove_blocked_by": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "spawn_teammate", "description": "Spawn a persistent autonomous teammate.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
    {"name": "shutdown_request", "description": "Request a teammate to shut down.",
     "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    {"name": "plan_approval", "description": "Approve or reject a teammate's plan.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}},
    {"name": "idle", "description": "Enter idle state.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "claim_task", "description": "Claim a task from the board.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
]


# ============================================================
# SECTION: agent_loop — 主智能体循环
# ============================================================

def agent_loop(messages: list):
    """
       主智能体的核心驱动循环。每次用户输入后调用一次，持续到 LLM 停止调用工具。

       【每轮执行顺序】
       1. microcompact     — 清除旧工具结果，节省 token
       2. auto_compact     — 若 token 超阈值则摘要压缩（会重置 messages）
       3. drain bg notifs  — 收集后台任务完成通知，注入到下一次 LLM 输入
       4. read inbox       — 读取其他 teammate 发给 lead 的消息
       5. LLM call         — 携带完整 messages + tools 调用 API
       6. tool dispatch    — 解析 tool_use block，分发到对应处理函数
       7. todo nag         — 若有未完成 todo 且已超过 3 轮未更新，追加提醒

       【退出条件】
       stop_reason != "tool_use"（LLM 不再调用工具，输出文本响应）
       或触发手动 compress（压缩后立即返回，等待下次用户输入）。
       """
    rounds_without_todo = 0  # 记录连续多少轮没有更新 TodoWrite
    while True:
        # s06: 微压缩（原地修改 messages，无返回值）
        microcompact(messages)

        # s06: 超阈值时自动压缩（messages 被替换为摘要）
        if estimate_tokens(messages) > TOKEN_THRESHOLD:
            print("[auto-compact triggered]")
            messages[:] = auto_compact(messages)

        # s08: 注入后台任务通知
        notifs = BG.drain()
        if notifs:
            txt = "\n".join(f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs)
            messages.append({"role": "user", "content": f"<background-results>\n{txt}\n</background-results>"})

        # s10: 检查收件箱消息（teammates 发给 lead 的消息）
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({"role": "user", "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>"})

        # 调用 LLM
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        # LLM 没有调用工具 → 已产生文本响应，退出循环
        if response.stop_reason != "tool_use":
            return

        # s2: 执行工具调用
        results = []
        used_todo = False
        manual_compress = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compress":
                    manual_compress = True  # 标记，等所有工具执行完再压缩
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                if block.name == "TodoWrite":
                    used_todo = True

        # s03: todo 提醒逻辑，连续 3 轮没有更新 todo 但仍有未完成项 → 追加提醒
        rounds_without_todo = 0 if used_todo else rounds_without_todo + 1
        if TODO.has_open_items() and rounds_without_todo >= 3:
            results.append({"type": "text", "text": "<reminder>Update your todos.</reminder>"})
        messages.append({"role": "user", "content": results})

        # s06: 手动压缩：执行完所有工具后才压缩，然后返回（等待用户下一轮输入）
        if manual_compress:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
            return


# ============================================================
# SECTION: repl — 用户交互入口
# ============================================================
if __name__ == "__main__":
    """
       命令行 REPL（Read-Eval-Print Loop）。

       【特殊命令】
       /compact — 手动触发对话压缩
       /tasks   — 查看所有持久任务（不触发 LLM）
       /team    — 查看团队成员状态（不触发 LLM）
       /inbox   — 查看并清空主智能体收件箱（不触发 LLM）

       【正常输入】
       任意文本 → 追加到 history → 调用 agent_loop → 打印 LLM 回复

       history 列表在整个会话中持续累积（实现多轮对话）。
       """
    history = []    # 完整对话历史（user + assistant 交替）
    while True:
        try:
            query = input("\033[36ms_full >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 特殊命令处理（不触发 LLM）
        if query.strip() == "/compact":
            if history:
                print("[manual compact via /compact]")
                history[:] = auto_compact(history)
            continue
        if query.strip() == "/tasks":
            print(TASK_MGR.list_all())
            continue
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue

        # 正常输入：加入对话历史，驱动智能体
        history.append({"role": "user", "content": query})
        agent_loop(history)

        # 打印最后一条 assistant 回复
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()