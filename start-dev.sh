#!/usr/bin/env bash
# =============================================================================
# start-dev.sh —— SkillForge 前后端一键启动脚本
# 用法：
#   ./start-dev.sh           启动前后端（单终端，Ctrl+C 全部停止）
#   QWEN_MODEL=qwen-plus NEXT_PUBLIC_CODEX_CHATGPT_MODE=0 ./start-dev.sh
#                            显式切回 Qwen 与旧 API Key 界面门槛
#   CODEX_REASONING_EFFORT=medium ./start-dev.sh  覆盖 Codex 推理档位
# 说明：端口已被占用时自动跳过对应服务；日志直接打印到当前终端。
# =============================================================================
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT=8891
FRONTEND_PORT=3000
# 临时默认使用本机 Codex ChatGPT 登录；Qwen、DeepSeek、GLM 仍可通过
# 既有 QWEN_MODEL / 角色模型变量显式启用。
QWEN_MODEL="${QWEN_MODEL:-gpt-5.6-sol}"
QWEN_ENABLE_THINKING="${QWEN_ENABLE_THINKING:-0}"
NEXT_PUBLIC_CODEX_CHATGPT_MODE="${NEXT_PUBLIC_CODEX_CHATGPT_MODE:-1}"

is_port_listening() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

PIDS=()

# ---- 后端 ---------------------------------------------------------------
if is_port_listening "$BACKEND_PORT"; then
  echo "[skillforge] 后端已在运行 (端口 $BACKEND_PORT)，跳过"
else
  echo "[skillforge] 启动后端 (端口 $BACKEND_PORT, 模型 $QWEN_MODEL, thinking=$QWEN_ENABLE_THINKING)..."
  (
    cd "$ROOT/backend" || exit 1
    exec env QWEN_ENABLE_THINKING="$QWEN_ENABLE_THINKING" QWEN_MODEL="$QWEN_MODEL" \
      ./.venv/bin/python app.py
  ) &
  PIDS+=("$!")
fi

# ---- 前端 ---------------------------------------------------------------
if is_port_listening "$FRONTEND_PORT"; then
  echo "[skillforge] 前端已在运行 (端口 $FRONTEND_PORT)，跳过"
else
  echo "[skillforge] 启动前端 (端口 $FRONTEND_PORT, Codex ChatGPT mode=$NEXT_PUBLIC_CODEX_CHATGPT_MODE)..."
  (
    cd "$ROOT/frontend" || exit 1
    exec env -u CODEBUDDY_SAFE_DELETE_BULK_STATE_DIR -u CODEBUDDY_TOOL_CALL_ID \
      NEXT_PUBLIC_CODEX_CHATGPT_MODE="$NEXT_PUBLIC_CODEX_CHATGPT_MODE" \
      npm run dev
  ) &
  PIDS+=("$!")
fi

cleanup() {
  echo
  echo "[skillforge] 停止所有服务..."
  for p in "${PIDS[@]}"; do
    kill "$p" 2>/dev/null
  done
  wait 2>/dev/null
  exit 0
}
trap cleanup INT TERM

echo "[skillforge] 就绪：前端 http://localhost:$FRONTEND_PORT ｜ 后端 http://127.0.0.1:$BACKEND_PORT"
echo "[skillforge] 按 Ctrl+C 停止全部服务"
wait
