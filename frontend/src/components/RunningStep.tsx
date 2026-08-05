"use client";

// =============================================================================
// 【文件头】RunningStep.tsx —— 第三步：运行优化并实时展示进度
// 职责：调用 /api/start 启动后台优化任务，然后每 3 秒轮询（polling）
//       /api/status 获取最新实验数据，用折线图与列表展示，完成后进入结果页。
// 接收：父组件传入的 sessionId / apiKey / scenarios / evals 与 onComplete。
// 输出：优化完成时调用 onComplete(finalResult) 上报结果。
// 建议先看：useEffect 启动时机 → startOptimization（开始 + 轮询循环）。
// 【初学者提示】"开始"只做一件事：POST /api/start 让后端把优化丢进后台任务，
//       然后立刻进入轮询循环 —— 进度不是一次性返回的，而是靠反复询问后端。
// =============================================================================

import { useState, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Loader2, StopCircle, ChevronDown, ChevronRight, GitCompareArrows } from "lucide-react";
import { startOptimization as apiStartOptimization, getStatus, stopOptimization } from "@/lib/api";

interface RunningStepProps {
  sessionId: string;
  apiKey: string;
  scenarios: any[];
  evals: any[];
  onComplete: (result: any) => void;
}

interface Candidate {
  candidate_id: number;
  description: string;
  score: number;
  reason: string;
}

interface Experiment {
  experiment_id: number;
  description: string;
  status: string;
  score: number;
  max_score: number;
  pass_rate: number;
  per_eval: any[];
  strategy?: string;
  candidates?: Candidate[];
  dimension_scores?: Record<string, { passed: number; total: number; pct: number }>;
  diff_summary?: string;
}

export default function RunningStep({
  sessionId,
  apiKey,
  scenarios,
  evals,
  onComplete,
}: RunningStepProps) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [currentScore, setCurrentScore] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [currentExperiment, setCurrentExperiment] = useState<string>("");
  const [started, setStarted] = useState(false);
  // 【C6 过程可视化】每个实验的 diff 折叠状态（key = experiment_id）。
  const [openDiffs, setOpenDiffs] = useState<Record<number, boolean>>({});

  // 【异步】useEffect 只在组件挂载时运行一次（依赖数组为空）：用 started 标记
  // 保证只触发一次 startOptimization，避免 React 严格模式下的重复启动。
  useEffect(() => {
    if (!started) {
      setStarted(true);
      startOptimization();
    }
  }, []);

  const startOptimization = async () => {
    setIsRunning(true);

    try {
      // Start the optimization
      // 【主流程】POST /api/start 时把 API Key 与轮数上限交给后端；后端立即
      // 返回 {"status":"started"} 并让优化在后台任务中运行。
      await apiStartOptimization(sessionId, apiKey, { max_rounds: 20 });

      // Poll for status every 3 seconds
      // 【主流程】轮询循环：每 3 秒问一次 /api/status；有新增实验就刷新界面，
      // 遇到 complete / error / stopped 三种终态分别处理。lastExpCount 用来
      // 判断"是否有新实验"（避免重复写入相同数据）。
      const poll = async () => {
        let lastExpCount = 0;
        while (true) {
          await new Promise((r) => setTimeout(r, 3000));

          try {
            const data = await getStatus(sessionId);

            // Update experiments if new ones arrived
            if (data.experiments && data.experiments.length > lastExpCount) {
              const newExps = data.experiments;
              setExperiments(newExps);
              const latest = newExps[newExps.length - 1];
              setCurrentScore(latest.pass_rate);
              setCurrentExperiment(
                latest.status === "baseline"
                  ? "Baseline complete"
                  : `Experiment ${latest.experiment_id}: ${latest.status}`
              );
              lastExpCount = newExps.length;
            }

            // Check if complete
            // 终态 1：complete + final_result → 上报结果并退出轮询。
            if (data.status === "complete" && data.final_result) {
              setIsRunning(false);
              onComplete(data.final_result);
              return;
            }

            // 终态 2：error → 弹窗提示并退出。
            if (data.status === "error") {
              setIsRunning(false);
              alert(`Error: ${data.error || "Unknown error"}`);
              return;
            }

            // 终态 3：stopped → 静默退出（用户点了停止）。
            if (data.status === "stopped") {
              setIsRunning(false);
              return;
            }
          } catch {
            // Network error, keep polling
            // 网络异常时继续轮询，不中断。
          }
        }
      };

      poll();
    } catch (error: any) {
      alert(error.message || "Failed to start optimization.");
      setIsRunning(false);
    }
  };

  // 【主流程】停止：通知后端 /api/stop。后端会协作式取消后台优化
  // （轮间检查 stop 标记；当次模型调用返回后生效）。
  const handleStop = async () => {
    try {
      await stopOptimization(sessionId);
      setIsRunning(false);
    } catch (error) {
      alert("Failed to stop optimization");
    }
  };

  // 派生图表数据：把 experiments 映射成折线图需要的 { experiment, passRate }，
  // 基线（experiment_id 0）在横轴上显示为 "Base"。
  const chartData = experiments.map((exp) => {
    // 【C6 逐维度折线】若该实验带 dimension_scores，则为每个维度生成一列。
    const dims: Record<string, number> = {};
    if (exp.dimension_scores) {
      Object.entries(exp.dimension_scores).forEach(([dim, s]) => {
        dims[`dim_${dim}`] = s.pct;
      });
    }
    return {
      experiment: exp.experiment_id === 0 ? "Base" : exp.experiment_id.toString(),
      passRate: exp.pass_rate,
      status: exp.status,
      ...dims,
    };
  });

  // 汇总出现过哪些维度（用于画多条维度线）。
  const dimensionKeys: string[] = [];
  experiments.forEach((exp) => {
    if (exp.dimension_scores) {
      Object.keys(exp.dimension_scores).forEach((dim) => {
        if (!dimensionKeys.includes(`dim_${dim}`)) dimensionKeys.push(`dim_${dim}`);
      });
    }
  });
  // 维度配色轮换（青绿品牌 + 邻近色）。
  const DIM_COLORS = ["#22d3ee", "#14b8a6", "#38bdf8", "#34d399", "#a78bfa"];

  // 派生评估明细：把最新一次实验的 per_eval 结果按 eval_id 对到每条评估标准上。
  const evalBreakdown = evals.map((evalItem) => {
    const latestExperiment = experiments[experiments.length - 1];
    if (!latestExperiment) {
      return { ...evalItem, passed: 0, total: 0, passRate: 0 };
    }

    const evalResult = latestExperiment.per_eval?.find(
      (pe: any) => pe.eval_id === evalItem.id
    );

    return {
      ...evalItem,
      passed: evalResult?.passed || 0,
      total: evalResult?.total || 0,
      passRate: evalResult?.pass_rate || 0,
    };
  });

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      <div className="glass rounded-2xl p-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold mb-2">
              {currentScore.toFixed(1)}
              <span className="text-lg text-zinc-500">%</span>
            </h2>
            <p className="text-zinc-400">{currentExperiment}</p>
          </div>

          {isRunning && (
            <button
              onClick={handleStop}
              className="px-6 py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-xl transition-all flex items-center gap-2"
            >
              <StopCircle className="w-5 h-5" />
              Stop Optimization
            </button>
          )}
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis
              dataKey="experiment"
              stroke="#71717a"
              style={{ fontSize: "12px" }}
            />
            <YAxis
              stroke="#71717a"
              style={{ fontSize: "12px" }}
              domain={[0, 100]}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#18181b",
                border: "1px solid #27272a",
                borderRadius: "8px",
              }}
            />
            <defs>
              <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
              </linearGradient>
            </defs>
            <Line
              type="monotone"
              dataKey="passRate"
              stroke="#22d3ee"
              strokeWidth={3}
              dot={(props: any) => {
                const { cx, cy, payload, index } = props;
                const color =
                  payload.status === "baseline"
                    ? "#3b82f6"
                    : payload.status === "keep"
                    ? "#22c55e"
                    : "#ef4444";
                return <circle key={`dot-${index}`} cx={cx} cy={cy} r={5} fill={color} />;
              }}
              fill="url(#colorGradient)"
            />
            {/* 【C6】逐维度折线：展示各评分维度随轮次的变化。 */}
            {dimensionKeys.map((dimKey, i) => (
              <Line
                key={dimKey}
                type="monotone"
                dataKey={dimKey}
                stroke={DIM_COLORS[i % DIM_COLORS.length]}
                strokeWidth={1.5}
                strokeDasharray="5 4"
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="glass rounded-2xl p-8">
          <h3 className="text-xl font-bold mb-6">Experiment Log</h3>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {experiments.length === 0 ? (
              <div className="flex items-center justify-center py-12 text-zinc-500">
                <Loader2 className="w-8 h-8 animate-spin" />
              </div>
            ) : (
              experiments.map((exp) => (
                <div
                  key={exp.experiment_id}
                  className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className={`
                        px-3 py-1 rounded-full text-xs font-medium
                        ${
                          exp.status === "baseline"
                            ? "bg-blue-500/20 text-blue-400"
                            : exp.status === "keep"
                            ? "bg-green-500/20 text-green-400"
                            : "bg-red-500/20 text-red-400"
                        }
                      `}
                      >
                        {exp.status === "baseline"
                          ? "Baseline"
                          : exp.status === "keep"
                          ? "Keep ✓"
                          : "Discard ✗"}
                      </span>
                      {/* 【C6】策略徽章：展示本实验使用的变异策略。 */}
                      {exp.strategy && (
                        <span className="px-2 py-1 rounded-full text-xs font-medium bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                          {exp.strategy}
                        </span>
                      )}
                      {/* 【C6】并行候选数：本轮尝试了几个变异方向。 */}
                      {exp.candidates && exp.candidates.length > 1 && (
                        <span className="px-2 py-1 rounded-full text-xs font-medium bg-zinc-800 text-zinc-400">
                          {exp.candidates.length} candidates
                        </span>
                      )}
                    </div>
                    <span className="text-lg font-bold text-cyan-400">
                      {exp.pass_rate}%
                    </span>
                  </div>
                  <p className="text-sm text-zinc-400">
                    {exp.experiment_id === 0
                      ? "Initial baseline"
                      : exp.description}
                  </p>

                  {/* 【C6】并行候选明细：每个候选的分数与取舍原因。 */}
                  {exp.candidates && exp.candidates.length > 1 && (
                    <div className="mt-3 space-y-1.5">
                      {exp.candidates.map((c) => (
                        <div
                          key={c.candidate_id}
                          className="flex items-center justify-between px-3 py-1.5 rounded-md bg-zinc-800/60 text-xs"
                        >
                          <span className="text-zinc-300 flex items-center gap-2">
                            <GitCompareArrows className="w-3.5 h-3.5 text-cyan-500" />
                            #{c.candidate_id} {c.description || "(no change)"}
                          </span>
                          <span className="flex items-center gap-2">
                            <span
                              className={c.reason === "best"
                                ? "text-green-400"
                                : c.reason === "not_best"
                                ? "text-zinc-400"
                                : "text-red-400"}
                            >
                              {c.score}%
                            </span>
                            <span className="text-zinc-500">{c.reason}</span>
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* 【C6】diff 折叠视图：保留轮次的 SKILL.md 变更明细。 */}
                  {exp.diff_summary && (
                    <div className="mt-3">
                      <button
                        onClick={() =>
                          setOpenDiffs((prev) => ({
                            ...prev,
                            [exp.experiment_id]: !prev[exp.experiment_id],
                          }))
                        }
                        className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-cyan-400 transition-colors"
                      >
                        {openDiffs[exp.experiment_id] ? (
                          <ChevronDown className="w-3.5 h-3.5" />
                        ) : (
                          <ChevronRight className="w-3.5 h-3.5" />
                        )}
                        View diff
                      </button>
                      {openDiffs[exp.experiment_id] && (
                        <pre className="mt-2 p-3 rounded-md bg-zinc-950 border border-zinc-800 text-[11px] leading-relaxed text-zinc-400 overflow-x-auto whitespace-pre-wrap break-words">
                          {exp.diff_summary}
                        </pre>
                      )}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        <div className="glass rounded-2xl p-8">
          <h3 className="text-xl font-bold mb-6">Evaluation Breakdown</h3>
          <div className="space-y-4">
            {evalBreakdown.map((evalItem) => (
              <div key={evalItem.id}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">{evalItem.name}</span>
                  <span className="text-sm text-zinc-400">
                    {evalItem.passed}/{evalItem.total}
                  </span>
                </div>
                <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className="h-full gradient-bg transition-all duration-500"
                    style={{ width: `${evalItem.passRate}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
