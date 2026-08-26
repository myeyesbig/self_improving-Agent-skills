// =============================================================================
// 【文件头】lib/api.ts —— 前端与后端通信的唯一出口
// 职责：集中管理 API_BASE 与全部端点的类型化调用，消除各组件里重复的
//       fetch 硬编码。组件只 import 这里的函数，不再各自拼 URL。
// 接收：各组件传入的参数（文件、session_id、apiKey、scenarios 等）。
// 输出：解析后的 JSON / Blob。
// 建议先看：API_BASE 与 downloadSkill 之外的各请求函数。
// 【初学者提示】所有函数都返回已解析的 data（失败时抛 Error），调用方
//       只需要 await 即可，不需要重复写 fetch + response.ok 检查。
// =============================================================================

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8891";

// 临时默认的 Codex ChatGPT 模式由本机登录态认证，仍保留 qwen_api_key
// 请求字段但允许空字符串。显式切回旧模型时设为 0，恢复原有前端 key 门槛。
export const CODEX_CHATGPT_MODE =
  process.env.NEXT_PUBLIC_CODEX_CHATGPT_MODE !== "0";

// -- 通用请求封装 -------------------------------------------------------------

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

// -- 会话与上传 ---------------------------------------------------------------

export interface SessionPayload {
  session_id: string;
  file_list: string[];
  metadata: any;
}

/** 上传 .zip 技能包（POST /api/upload） */
export async function uploadZip(file: File): Promise<SessionPayload> {
  const formData = new FormData();
  formData.append("file", file);
  return request<SessionPayload>(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  });
}

/** 上传文件夹中的多文件（POST /api/upload-files） */
export async function uploadMultipleFiles(files: File[]): Promise<SessionPayload> {
  const formData = new FormData();
  for (const f of files) {
    const path = (f as any).webkitRelativePath || f.name;
    formData.append("files", f, path);
  }
  return request<SessionPayload>(`${API_BASE}/api/upload-files`, {
    method: "POST",
    body: formData,
  });
}

// -- 分析 / 配置 --------------------------------------------------------------

export interface AnalyzePayload {
  scenarios: any[];
  evals: any[];
}

/** 让当前配置的模型分析技能并生成测试场景与评估标准（POST /api/analyze） */
export async function analyzeSkill(
  sessionId: string,
  apiKey: string,
  deepseekApiKey?: string
): Promise<AnalyzePayload> {
  return request<AnalyzePayload>(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      qwen_api_key: apiKey,
      ...(deepseekApiKey ? { deepseek_api_key: deepseekApiKey } : {}),
    }),
  });
}

/** 重新生成 scenarios/evals（POST /api/regenerate） */
export async function regenerateConfig(
  sessionId: string,
  apiKey: string,
  deepseekApiKey?: string
): Promise<AnalyzePayload> {
  return request<AnalyzePayload>(`${API_BASE}/api/regenerate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      qwen_api_key: apiKey,
      ...(deepseekApiKey ? { deepseek_api_key: deepseekApiKey } : {}),
    }),
  });
}

/** 保存用户勾选/编辑后的配置（POST /api/update-config） */
export async function updateConfig(
  sessionId: string,
  scenarios: any[],
  evals: any[]
): Promise<{ status: string }> {
  return request<{ status: string }>(`${API_BASE}/api/update-config`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, scenarios, evals }),
  });
}

// -- 优化运行 -----------------------------------------------------------------

export interface StartOptions {
  max_rounds?: number;
  parallel_mutations?: number;
  strategy_pool?: string[];
  improvement_threshold?: number;
}

/** 启动后台优化任务（POST /api/start/{sessionId}） */
export async function startOptimization(
  sessionId: string,
  apiKey: string,
  options: StartOptions = {},
  deepseekApiKey?: string
): Promise<{ status: string }> {
  const body: Record<string, any> = {
    qwen_api_key: apiKey,
    ...(deepseekApiKey ? { deepseek_api_key: deepseekApiKey } : {}),
    max_rounds: options.max_rounds ?? 20,
  };
  if (options.parallel_mutations !== undefined) body.parallel_mutations = options.parallel_mutations;
  if (options.strategy_pool !== undefined) body.strategy_pool = options.strategy_pool;
  if (options.improvement_threshold !== undefined) body.improvement_threshold = options.improvement_threshold;
  return request<{ status: string }>(`${API_BASE}/api/start/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export interface StatusPayload {
  status: string;
  experiments: any[];
  error?: string | null;
  final_result?: any;
}

/** 轮询优化进度（GET /api/status/{sessionId}） */
export async function getStatus(sessionId: string): Promise<StatusPayload> {
  return request<StatusPayload>(`${API_BASE}/api/status/${sessionId}`);
}

/** 请求停止优化（POST /api/stop/{sessionId}） */
export async function stopOptimization(sessionId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`${API_BASE}/api/stop/${sessionId}`, {
    method: "POST",
  });
}

// -- 示例技能 -----------------------------------------------------------------

export interface ExampleSkill {
  name: string;
  description: string;
  path: string;
}

/** 获取可用示例技能列表（GET /api/examples） */
export async function listExamples(): Promise<ExampleSkill[]> {
  const data = await request<{ examples: ExampleSkill[] }>(`${API_BASE}/api/examples`);
  return data.examples;
}

/** 加载示例技能为会话（POST /api/examples/{path}/load） */
export async function loadExample(path: string): Promise<SessionPayload> {
  return request<SessionPayload>(`${API_BASE}/api/examples/${encodeURIComponent(path)}/load`, {
    method: "POST",
  });
}

// -- 下载 ---------------------------------------------------------------------

/** 下载改进后的技能包 zip（GET /api/download/{sessionId}），返回 Blob */
export async function downloadSkill(sessionId: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/api/download/${sessionId}`);
  if (!response.ok) {
    throw new Error("Download failed");
  }
  return response.blob();
}
