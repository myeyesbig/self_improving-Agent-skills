# =============================================================================
# 【文件头】app.py —— FastAPI 后端入口（HTTP 路由 + 内存会话管理）
# 职责：定义全部 API 路由（上传 / 分析 / 配置 / 启动优化 / 停止 / 轮询状态 /
#       SSE 流 / 下载），并用进程内字典 sessions 保存每个会话的状态。
# 接收：浏览器发来的 .zip 文件 / 多个文件（文件夹上传）、JSON 请求体（含
#       qwen_api_key 与 session_id）。
# 输出：JSON 响应体（session_id、scenarios、evals、status、final_result 等）、
#       SSE 事件流、打包好的 improved_skill.zip。
# 建议先看：/api/upload → /api/analyze → /api/start → /api/status 这条主链路，
#       以及 start_optimization() 里的后台任务与闭包 callback。
# 【注意】前端实际使用 /api/status 轮询（polling）获取进度；/api/stream 的
#       SSE 路由虽然存在，但当前 UI 并未消费它。另外 /api/stop 只把 session
#       状态改为 stopped 并关闭事件队列，并不会真正取消后台正在跑的模型调用。
# =============================================================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import time
import uuid
import zipfile
import io
import os
import json
import asyncio
import tempfile
import shutil
import re
import logging
import traceback
from qwen_optimizer import SkillOptimizer, StopOptimizationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 【安全】上传与会话的硬性限制：总量 10MB、单文件 1MB、最多 50 个文件、
# 会话 1 小时过期；只允许白名单里的文本扩展名。
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB total
MAX_FILE_SIZE = 1 * 1024 * 1024     # 1 MB per file
MAX_FILE_COUNT = 50
SESSION_TTL = 3600  # 1 hour
ALLOWED_EXTENSIONS = {
    ".md", ".txt", ".json", ".yaml", ".yml", ".py", ".js", ".ts",
    ".html", ".css", ".xml", ".toml", ".cfg", ".ini", ".sh",
}

app = FastAPI()

# CORS：允许所有来源访问 API，便于前端开发时跨端口调用。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 【主流程】内存会话表：session_id → 该次上传的完整状态。所有路由都读写这个
# 字典，进程重启即清空；session 字段的具体含义见 CODE_WALKTHROUGH.zh-CN.md。
sessions: Dict[str, dict] = {}


# -- 请求体模型 ---------------------------------------------------------------
# 【兼容】请求字段名保持 qwen_api_key。默认 gpt- 路由使用本机 ChatGPT 登录，
# 因而允许空字符串；覆盖为 Qwen/DeepSeek/GLM 时仍按既有规则使用相应凭据。
# 密钥只随请求体传到内存，绝不持久化。

class AnalyzeRequest(BaseModel):
    session_id: str
    qwen_api_key: str
    deepseek_api_key: Optional[str] = None


class SessionConfig(BaseModel):
    session_id: str
    scenarios: List[dict]
    evals: List[dict]


class RegenerateRequest(BaseModel):
    session_id: str
    qwen_api_key: str
    deepseek_api_key: Optional[str] = None


class StartRequest(BaseModel):
    qwen_api_key: str
    # 【双 key】可选 DeepSeek key；未传时 DeepSeek 分支回退到
    # qwen_api_key / DEEPSEEK_API_KEY 环境变量。
    deepseek_api_key: Optional[str] = None
    max_rounds: Optional[int] = Field(default=20, gt=0, le=50)
    # 【C3】并行变异数：可选，默认走后端配置（通常为 1）。
    parallel_mutations: Optional[int] = Field(default=None, ge=1, le=3)
    # 【C1】策略白名单：可选，默认全量模板。
    strategy_pool: Optional[List[str]] = None
    # 【C4】提升阈值：可选，默认 0.0（严格高于）。
    improvement_threshold: Optional[float] = Field(default=None, ge=0.0)


def parse_skill_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from SKILL.md"""
    # 简单解析 SKILL.md 顶部的 YAML frontmatter（--- 分隔的键值对），
    # 用来提取技能名称与描述，展示在界面上；解析失败就返回空 dict。
    # 支持 `>-` / `|-` / `>` / `|` 折叠块：后续缩进行的文本会拼成单段描述。
    if not content.startswith("---"):
        return {}
    try:
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}
        frontmatter = parts[1].strip()
        metadata = {}
        lines = frontmatter.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if ":" in line and not line.startswith(" "):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"')
                # 折叠块指示符：收集后续以 2+ 空格缩进的行，折叠成单段文本。
                if value in (">-", "|-", ">", "|", ">- ", "|- "):
                    parts_list = []
                    i += 1
                    while i < len(lines):
                        cont = lines[i]
                        if cont.startswith("  ") or cont.startswith("\t"):
                            parts_list.append(cont.strip())
                            i += 1
                        else:
                            break
                    metadata[key] = " ".join(parts_list).strip()
                    continue
                metadata[key] = value
            i += 1
        return metadata
    except Exception:
        return {}


def create_session_from_files(skill_files: dict, file_list: list) -> dict:
    """Create a session dict from skill files"""
    # 【主流程】上传成功的关键一步：生成 session_id，并把本次上传的技能文件
    # 与初始状态存进 sessions 表。之后所有路由都凭 session_id 找到这份状态。
    skill_md = None
    for name, content in skill_files.items():
        if name.endswith("SKILL.md"):
            skill_md = content
            break
    if not skill_md:
        raise HTTPException(status_code=400, detail="No SKILL.md found")
    metadata = parse_skill_frontmatter(skill_md)
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "skill_files": skill_files,
        "file_list": file_list,
        "metadata": metadata,
        "status": "uploaded",
        "scenarios": None,
        "evals": None,
        "experiments": [],
        "changelog": [],
        "current_skill_md": skill_md,
        "original_skill_md": skill_md,
        "created_at": time.time(),
    }
    # 返回给前端的只有 session_id 和文件清单；skill_files 等内部数据不出内存。
    return {"session_id": session_id, "file_list": file_list, "metadata": metadata}


def _is_allowed_file(name: str) -> bool:
    """Check if file extension is in the allowed text-file list."""
    _, ext = os.path.splitext(name)
    return ext.lower() in ALLOWED_EXTENSIONS


def _is_safe_path(name: str) -> bool:
    """Reject path traversal attempts."""
    # 【安全】拒绝路径穿越（..）与绝对路径，防止恶意压缩包把文件写到任意位置。
    return ".." not in name and not os.path.isabs(name)


# =============================================================================
# 路由：上传 → 分析 → 配置 → 优化 → 结果，见 CODE_WALKTHROUGH.zh-CN.md 的链路图
# =============================================================================

@app.post("/api/upload")
async def upload_skill(file: UploadFile = File(...)):
    """Accept zip file or multiple files, extract, return file list + parsed SKILL.md metadata"""
    # 【主流程】步骤 1a：接收 .zip，逐条检查大小 / 路径 / 扩展名后解压成文本，
    # 最后创建 session 并把 session_id 返回给前端。
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_SIZE // (1024*1024)}MB limit")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            skill_files = {}
            file_list = []
            for name in zf.namelist():
                # 跳过目录条目、macOS 残留与隐藏文件，再依次做路径、扩展名、大小校验。
                if name.endswith("/") or name.startswith("__MACOSX") or "/.DS_Store" in name or name.endswith(".DS_Store"):
                    continue
                if not _is_safe_path(name):
                    logger.warning(f"Skipping unsafe zip entry: {name}")
                    continue
                if not _is_allowed_file(name):
                    logger.info(f"Skipping non-text file: {name}")
                    continue
                raw = zf.read(name)
                if len(raw) > MAX_FILE_SIZE:
                    logger.warning(f"Skipping oversized file: {name} ({len(raw)} bytes)")
                    continue
                if len(file_list) >= MAX_FILE_COUNT:
                    logger.warning("Max file count reached, skipping remaining entries")
                    break
                file_content = raw.decode("utf-8", errors="ignore")
                skill_files[name] = file_content
                file_list.append(name)

            # Normalize paths: strip common prefix directory
            # 去掉所有文件共有的顶层目录前缀，让路径以 SKILL.md 为根，后续一致。
            if file_list:
                common = os.path.commonpath(file_list)
                if common and common != file_list[0]:
                    skill_files = {os.path.relpath(k, common): v for k, v in skill_files.items()}
                    file_list = [os.path.relpath(f, common) for f in file_list]

            return create_session_from_files(skill_files, file_list)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error processing file")


@app.post("/api/upload-files")
async def upload_files(files: List[UploadFile] = File(...)):
    """Accept multiple files (folder upload via webkitdirectory)"""
    # 【主流程】步骤 1b：文件夹上传（浏览器 webkitdirectory 属性）走这个路由，
    # 校验逻辑与 ZIP 上传一致，只是输入是文件列表而非压缩包。
    if len(files) > MAX_FILE_COUNT:
        raise HTTPException(status_code=413, detail=f"Too many files (max {MAX_FILE_COUNT})")
    skill_files = {}
    file_list = []
    total_size = 0
    for f in files:
        if f.filename.startswith(".") or "/.DS_Store" in (f.filename or "") or "__MACOSX" in (f.filename or ""):
            continue
        name = f.filename or "unknown"
        if not _is_safe_path(name):
            logger.warning(f"Skipping unsafe path: {name}")
            continue
        if not _is_allowed_file(name):
            logger.info(f"Skipping non-text file: {name}")
            continue
        content = await f.read()
        total_size += len(content)
        if total_size > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"Total upload exceeds {MAX_UPLOAD_SIZE // (1024*1024)}MB limit")
        if len(content) > MAX_FILE_SIZE:
            logger.warning(f"Skipping oversized file: {name} ({len(content)} bytes)")
            continue
        skill_files[name] = content.decode("utf-8", errors="ignore")
        file_list.append(name)

    # Normalize paths: strip common prefix directory
    if file_list:
        common = os.path.commonpath(file_list)
        if common and common != file_list[0]:
            skill_files = {os.path.relpath(k, common): v for k, v in skill_files.items()}
            file_list = [os.path.relpath(f, common) for f in file_list]

    return create_session_from_files(skill_files, file_list)


@app.post("/api/analyze")
async def analyze_skill(request: AnalyzeRequest):
    """Generate scenarios and evals using the configured model route."""
    # 【主流程】步骤 2：用 SkillOptimizer 的 Executor 助手分析技能文件，
    # 生成测试场景与评估标准，写回 session，前端据此进入配置页。
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]
    try:
        optimizer = SkillOptimizer(
            api_key=request.qwen_api_key,
            deepseek_api_key=request.deepseek_api_key,
        )
        analysis = await optimizer.analyze_skill(session["skill_files"])
        session["scenarios"] = analysis["scenarios"]
        session["evals"] = analysis["evals"]
        session["domain"] = analysis.get("domain", "other")
        session["status"] = "analyzed"
        return {"scenarios": analysis["scenarios"], "evals": analysis["evals"], "domain": session["domain"]}
    except Exception as e:
        logger.error(f"Analysis error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check your Codex login or provider credentials and try again.",
        )


@app.post("/api/regenerate")
async def regenerate_config(request: RegenerateRequest):
    """Regenerate scenarios/evals for a session"""
    # 复用 analyze_skill 逻辑：重新生成一份 scenarios/evals，覆盖原配置。
    analyze_req = AnalyzeRequest(
        session_id=request.session_id,
        qwen_api_key=request.qwen_api_key,
        deepseek_api_key=request.deepseek_api_key,
    )
    return await analyze_skill(analyze_req)


@app.post("/api/update-config")
async def update_config(config: SessionConfig):
    """Save user's selected/edited scenarios + evals"""
    # 【主流程】步骤 3：保存用户在配置页勾选 / 编辑后的 scenarios 与 evals。
    if config.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[config.session_id]
    session["scenarios"] = config.scenarios
    session["evals"] = config.evals
    session["status"] = "configured"
    return {"status": "ok"}


@app.get("/api/stream/{session_id}")
async def stream_progress(session_id: str):
    """SSE endpoint streaming optimization progress"""
    # 【注意】SSE（Server-Sent Events）路由：从事件队列里取事件，按 SSE 格式
    # 推送给浏览器。但当前前端 UI 走的是 /api/status 轮询，这个路由尚未被消费。
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]

    async def event_generator():
        # 【异步】事件队列不存在时惰性创建；None 是结束哨兵，收到即退出循环。
        if "event_queue" not in session:
            session["event_queue"] = asyncio.Queue()
        queue = session["event_queue"]
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if "event_queue" in session:
                del session["event_queue"]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/api/start/{session_id}")
async def start_optimization(session_id: str, request: StartRequest):
    """Start optimization in background task"""
    # 【主流程】步骤 4：校验会话已配置完成后，把优化任务交给后台运行并立即
    # 返回 {"status": "started"}，前端随后通过 /api/status 轮询进度。
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    if not session.get("scenarios") or not session.get("evals"):
        raise HTTPException(status_code=400, detail="Must configure scenarios and evals first")
    if session.get("status") == "running":
        raise HTTPException(status_code=400, detail="Optimization already running")

    session["status"] = "running"
    session["stop_requested"] = False
    # Pre-create the event queue so events aren't lost before SSE connects
    # 预先创建事件队列，避免后续事件在 SSE 连上之前丢失（同时被 /api/status 复用）。
    session["event_queue"] = asyncio.Queue()
    qwen_key = request.qwen_api_key

    async def run_optimization():
        # 【异步】后台任务：这里是用 asyncio.create_task 启动的协程，不阻塞
        # 当前请求。qwen_key 来自请求体，只在这个闭包内使用。
        logger.info(f"Starting optimization for session {session_id}")
        # 【P0 修复】把请求里的提升阈值传给优化器构造参数，让前端配置真正生效；
        # None 时优化器内部仍读 IMPROVEMENT_THRESHOLD 环境变量（默认 0.0）。
        # 【C5 停止】stop_provider 把 session 的 stop_requested 实时暴露给优化器：
        # /api/stop 置位后，下一轮 check_stop 即抛 StopOptimizationError，
        # 协作式停止真正生效（轮间语义不变，在途模型调用完成）。
        optimizer = SkillOptimizer(
            api_key=qwen_key,
            improvement_threshold=request.improvement_threshold,
            stop_provider=lambda: session.get("stop_requested", False),
            deepseek_api_key=request.deepseek_api_key,
        )

        # 【主流程】闭包 callback：优化器的每次 emit 都会回到这里，把事件
        # 写入事件队列，同时把 baseline / experiment_result / complete 事件
        # 转换为前端轮询能读到的 session 字段（experiments、final_result）。
        async def callback(event):
            logger.info(f"Callback event: {event['type']}")
            if "event_queue" in session:
                await session["event_queue"].put(event)
            if event["type"] == "baseline":
                session["experiments"].append({
                    "experiment_id": 0,
                    "pass_rate": event["data"].get("score", 0),
                    "status": "baseline",
                    "per_eval": event["data"].get("per_eval", []),
                    "dimension_scores": event["data"].get("dimension_scores", {}),
                })
            elif event["type"] == "experiment_result":
                session["experiments"].append({
                    "experiment_id": event["data"].get("round", len(session["experiments"])),
                    "pass_rate": event["data"].get("score", 0),
                    "status": "keep" if event["data"].get("kept") else "discard",
                    "per_eval": event["data"].get("per_eval", []),
                    "description": event["data"].get("description", ""),
                    "strategy": event["data"].get("strategy", ""),
                    "candidates": event["data"].get("candidates", []),
                    "dimension_scores": event["data"].get("dimension_scores", {}),
                    "diff_summary": event["data"].get("diff_summary", ""),
                })
            elif event["type"] == "complete":
                session["status"] = "complete"
                data = event["data"]
                ml = data.get("mutation_log", [])
                # Transform to match frontend ResultsStep expectations
                # 把优化器的返回结构转换成前端 ResultsStep 期望的 final_result 形状。
                session["final_result"] = {
                    "baseline_score": data.get("baseline_score", 0),
                    "final_score": data.get("final_score", 0),
                    "improved_skill_md": data.get("improved_skill_md", ""),
                    "original_skill_md": session.get("original_skill_md", ""),
                    "score_history": data.get("score_history", []),
                    "experiments_run": len(ml),
                    "kept": sum(1 for m in ml if m.get("kept")),
                    "discarded": sum(1 for m in ml if not m.get("kept")),
                    "changelog": [
                        {
                            "description": m.get("description", m.get("diagnosis", "")),
                            "reasoning": m.get("diagnosis", ""),
                            "status": "keep" if m.get("kept") else "discard",
                            "score_before": m.get("score_before", 0),
                            "score_after": m.get("score_after", 0),
                            "strategy": m.get("strategy_type", ""),
                        }
                        for m in ml
                    ],
                    "mutation_log": ml,
                    "strategy_stats": data.get("strategy_stats", {}),
                }
                session["current_skill_md"] = data.get("improved_skill_md", "")
                # 结束哨兵：通知 SSE 生成器退出。
                if "event_queue" in session:
                    await session["event_queue"].put(None)

        try:
            result = await optimizer.optimize(
                skill_files=session["skill_files"],
                scenarios=session["scenarios"],
                evals=session["evals"],
                max_rounds=request.max_rounds,
                callback=callback,
                parallel_mutations=request.parallel_mutations,
                strategy_pool=request.strategy_pool,
                domain=session.get("domain"),
                # 【阶段 5 断点续跑】thread_id=session_id：SKILL_CHECKPOINT_FILE
                # 设置时按会话隔离 checkpoint，重启/停止后可续跑同一优化任务。
                thread_id=session_id,
            )
            logger.info(f"Optimization complete: {result['baseline_score']}% -> {result['final_score']}%")
            # Don't overwrite final_result if callback already set it with transformed data
            # callback 已写入完整 final_result 时不再覆盖；这里是兜底分支。
            if not session.get("final_result"):
                ml = result.get("mutation_log", [])
                session["final_result"] = {
                    "baseline_score": result.get("baseline_score", 0),
                    "final_score": result.get("final_score", 0),
                    "improved_skill_md": result.get("improved_skill_md", ""),
                    "original_skill_md": session.get("original_skill_md", ""),
                    "score_history": result.get("score_history", []),
                    "experiments_run": len(ml),
                    "kept": sum(1 for m in ml if m.get("kept")),
                    "discarded": sum(1 for m in ml if not m.get("kept")),
                    "changelog": [
                        {
                            "description": m.get("description", m.get("diagnosis", "")),
                            "reasoning": m.get("diagnosis", ""),
                            "status": "keep" if m.get("kept") else "discard",
                            "score_before": m.get("score_before", 0),
                            "score_after": m.get("score_after", 0),
                            "strategy": m.get("strategy_type", ""),
                        }
                        for m in ml
                    ],
                    "mutation_log": ml,
                }
            session["current_skill_md"] = result["improved_skill_md"]
            session["status"] = "complete"
        except StopOptimizationError:
            # 【C5 停止】用户请求停止：协作式取消，在轮间生效。
            logger.info(f"Optimization stopped by user for session {session_id}")
            session["status"] = "stopped"
            if "event_queue" in session:
                await session["event_queue"].put({"type": "stopped", "data": {}})
                await session["event_queue"].put(None)
        except Exception as e:
            logger.error(f"Optimization error: {traceback.format_exc()}")
            session["status"] = "error"
            # Never surface the API key in error payloads.
            # 【安全】任何异常信息里如果包含 API Key，一律替换为占位符再返回。
            error_message = str(e)
            if qwen_key:
                error_message = error_message.replace(qwen_key, "[REDACTED]")
            session["error"] = error_message
            if "event_queue" in session:
                await session["event_queue"].put({"type": "error", "data": {"message": error_message}})
                await session["event_queue"].put(None)

    asyncio.create_task(run_optimization())
    return {"status": "started"}


@app.post("/api/stop/{session_id}")
async def stop_optimization(session_id: str):
    """Stop optimization"""
    # 【注意】停止接口的真实行为：只把 session 状态改为 stopped、放入结束哨兵
    # 关闭事件队列。后台的模型调用仍在跑，不会被真正取消（属当前实现边界）。
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    session["stop_requested"] = True
    session["status"] = "stopped"
    if "event_queue" in session:
        await session["event_queue"].put(None)
    return {"status": "stopped"}


@app.get("/api/download/{session_id}")
async def download_skill(session_id: str):
    """Download improved skill as zip"""
    # 【主流程】步骤 5：把改进后的 SKILL.md 与其余原始文件重新打包成 zip，
    # 附带 CHANGELOG.json；临时文件 60 秒后由后台任务清理。
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    if not session.get("current_skill_md"):
        raise HTTPException(status_code=400, detail="No improved skill available")
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "improved_skill.zip")
    try:
        with zipfile.ZipFile(zip_path, "w") as zf:
            for filename, content in session["skill_files"].items():
                if filename.endswith("SKILL.md"):
                    zf.writestr(filename, session["current_skill_md"])
                else:
                    zf.writestr(filename, content)
            if session.get("final_result"):
                changelog_content = json.dumps(session["final_result"]["changelog"], indent=2)
                zf.writestr("CHANGELOG.json", changelog_content)
        return FileResponse(zip_path, media_type="application/zip", filename="improved_skill.zip")
    finally:
        asyncio.create_task(cleanup_temp_dir(temp_dir))


async def cleanup_temp_dir(temp_dir: str):
    """延迟 60 秒删除临时下载目录，避免影响正在发送的文件响应。"""
    await asyncio.sleep(60)
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


@app.get("/api/examples")
async def list_examples():
    """List available example skills"""
    # 扫描 skill-examples/*.zip 里的 SKILL.md frontmatter，作为"示例技能"
    # 展示给用户。示例包与上传走同一套校验管线，zip 内路径不会落盘。
    examples_dir = os.path.join(os.path.dirname(__file__), "..", "skill-examples")
    examples = []
    if os.path.isdir(examples_dir):
        for fname in sorted(os.listdir(examples_dir)):
            if not fname.endswith(".zip"):
                continue
            zip_path = os.path.join(examples_dir, fname)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    skill_md_name = next(
                        (n for n in zf.namelist() if n.endswith("SKILL.md")),
                        None,
                    )
                    if skill_md_name is None:
                        continue
                    content = zf.read(skill_md_name).decode("utf-8", errors="ignore")
                    metadata = parse_skill_frontmatter(content)
                    # path 用 zip 文件名（去 .zip 后缀），与 load_example 对应。
                    examples.append({
                        "name": metadata.get("name", fname[:-4]),
                        "description": metadata.get("description", ""),
                        "path": fname[:-4],
                    })
            except zipfile.BadZipFile:
                logger.warning(f"Skipping invalid example zip: {fname}")
                continue
    return {"examples": examples}


@app.post("/api/examples/{example_name}/load")
async def load_example(example_name: str):
    """Load an example skill as if it were uploaded"""
    # 【安全】示例名做白名单正则校验，防止路径穿越。
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", example_name):
        raise HTTPException(status_code=400, detail="Invalid example name")
    zip_path = os.path.join(
        os.path.dirname(__file__), "..", "skill-examples", f"{example_name}.zip"
    )
    if not os.path.isfile(zip_path):
        raise HTTPException(status_code=404, detail="Example skill not found")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            skill_files = {}
            file_list = []
            for name in zf.namelist():
                # 与 /api/upload 相同的校验管线：跳过目录/macOS 残留/隐藏文件，
                # 再做路径穿越、扩展名、大小校验。
                if name.endswith("/") or name.startswith("__MACOSX") or "/.DS_Store" in name or name.endswith(".DS_Store"):
                    continue
                if not _is_safe_path(name):
                    logger.warning(f"Skipping unsafe zip entry: {name}")
                    continue
                if not _is_allowed_file(name):
                    logger.info(f"Skipping non-text file: {name}")
                    continue
                raw = zf.read(name)
                if len(raw) > MAX_FILE_SIZE:
                    logger.warning(f"Skipping oversized file: {name} ({len(raw)} bytes)")
                    continue
                if len(file_list) >= MAX_FILE_COUNT:
                    break
                skill_files[name] = raw.decode("utf-8", errors="ignore")
                file_list.append(name)
            if not skill_files:
                raise HTTPException(status_code=400, detail="Example contains no usable files")
            return create_session_from_files(skill_files, file_list)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid example zip")


@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    """Poll-based status endpoint. Returns all experiments so far."""
    # 【主流程】前端轮询（polling）接口：每 3 秒调用一次，返回当前 status、
    # 已有的 experiments 列表、错误信息与最终结果。这是当前 UI 实际使用的
    # 进度获取方式（SSE 路由存在但未被消费）。
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    return {
        "status": session.get("status", "unknown"),
        "experiments": session.get("experiments", []),
        "error": session.get("error"),
        "final_result": session.get("final_result"),
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


async def _cleanup_expired_sessions():
    """Periodically remove sessions older than SESSION_TTL."""
    # 【注意】会话清理后台任务：每 5 分钟删除超过 1 小时的会话；运行中的
    # 会话（running）不会被清理，避免打断正在进行的长任务。
    while True:
        await asyncio.sleep(300)  # every 5 minutes
        now = time.time()
        expired = [
            sid for sid, s in sessions.items()
            if now - s.get("created_at", now) > SESSION_TTL
            and s.get("status") not in ("running",)
        ]
        for sid in expired:
            del sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired session(s)")


@app.on_event("startup")
async def startup():
    # 应用启动时拉起会话清理任务。
    asyncio.create_task(_cleanup_expired_sessions())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8891)
