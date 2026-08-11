"use client";

// =============================================================================
// 【文件头】page.tsx —— 首页与四步流程的"总指挥"
// 职责：管理 1→2→3→4 四步（上传 → 配置 → 优化 → 结果）的页面状态机，
//       保存跨步骤共享的数据（sessionId / apiKey / scenarios / evals / 结果）。
// 接收：各子组件通过 props 传入的回调（onComplete 等）上报结果。
// 输出：根据 currentStep 渲染对应的步骤组件。
// 建议先看：currentStep 与 handleUploadComplete → handleConfigComplete →
//       handleOptimizationComplete 这条"回调推动步骤前进"的链路。
// 【初学者提示】这是"父组件持有共享状态、子组件通过回调上报"的典型写法：
//       子组件不自己保存跨步骤数据，而是调用父组件给的函数把数据"交上去"。
// =============================================================================

import { useState } from "react";
import StepIndicator from "@/components/StepIndicator";
import UploadStep from "@/components/UploadStep";
import ConfigStep from "@/components/ConfigStep";
import RunningStep from "@/components/RunningStep";
import ResultsStep from "@/components/ResultsStep";
import Logo from "@/components/Logo";

export default function Home() {
  // 【主流程】四步状态机的核心：currentStep 决定渲染哪个步骤（1 上传 /
  // 2 配置 / 3 优化 / 4 结果）；其余 state 是跨步骤共享的"会话上下文"。
  const [currentStep, setCurrentStep] = useState(1);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [deepseekApiKey, setDeepseekApiKey] = useState("");
  const [metadata, setMetadata] = useState<any>(null);
  const [scenarios, setScenarios] = useState<any[]>([]);
  const [evals, setEvals] = useState<any[]>([]);
  const [finalResult, setFinalResult] = useState<any>(null);

  // 【主流程】回调 1：上传 + 分析完成后，把 sessionId / 两个 key / 元数据 /
  // 初始 scenarios / evals 存进父组件 state，并跳到步骤 2（配置）。
  const handleUploadComplete = (
    sid: string,
    key: string,
    meta: any,
    initialScenarios: any[],
    initialEvals: any[],
    deepKey?: string
  ) => {
    setSessionId(sid);
    setApiKey(key);
    setDeepseekApiKey(deepKey || "");
    setMetadata(meta);
    setScenarios(initialScenarios);
    setEvals(initialEvals);
    setCurrentStep(2);
  };

  // 【主流程】回调 2：配置页保存完成后跳到步骤 3（优化）。
  const handleConfigComplete = () => {
    setCurrentStep(3);
  };

  // 【主流程】回调 3：优化完成后保存结果并跳到步骤 4（结果页）。
  const handleOptimizationComplete = (result: any) => {
    setFinalResult(result);
    setCurrentStep(4);
  };

  // 重新开始：把所有共享状态清空，回到步骤 1。
  const handleStartOver = () => {
    setCurrentStep(1);
    setSessionId(null);
    setApiKey("");
    setDeepseekApiKey("");
    setMetadata(null);
    setScenarios([]);
    setEvals([]);
    setFinalResult(null);
  };

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Logo size={48} />
            <h1 className="text-5xl font-bold">
              <span className="gradient-text">Skill</span>Forge
            </h1>
          </div>
          <p className="text-zinc-400 dark:text-zinc-400 text-zinc-600 text-lg">
            Self-forging agent skills with Qwen — the 3-agent loop that hammers raw SKILL.md into production-grade tools
          </p>
          <div className="flex items-center justify-center gap-2 mt-3">
            <span className="px-3 py-1 bg-blue-500/10 text-blue-400 dark:text-blue-400 text-blue-600 rounded-full text-xs font-medium border border-blue-500/20">Qwen-Agent</span>
            <span className="px-3 py-1 bg-cyan-500/10 text-cyan-400 dark:text-cyan-400 text-cyan-600 rounded-full text-xs font-medium border border-cyan-500/20">Qwen</span>
            <span className="px-3 py-1 bg-emerald-500/10 text-emerald-400 dark:text-emerald-400 text-emerald-600 rounded-full text-xs font-medium border border-emerald-500/20">Multi-Agent</span>
          </div>
        </div>

        <StepIndicator currentStep={currentStep} />

        <div className="mt-12">
          {/* 【主流程】条件渲染：currentStep 决定显示哪一步。注意步骤 2/3/4
              都依赖 sessionId 存在，所以加了 && sessionId 守卫。 */}
          {currentStep === 1 && (
            <UploadStep onComplete={handleUploadComplete} />
          )}

          {currentStep === 2 && sessionId && (
            <ConfigStep
              sessionId={sessionId}
              apiKey={apiKey}
              deepseekApiKey={deepseekApiKey}
              scenarios={scenarios}
              evals={evals}
              onScenariosChange={setScenarios}
              onEvalsChange={setEvals}
              onComplete={handleConfigComplete}
            />
          )}

          {currentStep === 3 && sessionId && (
            <RunningStep
              sessionId={sessionId}
              apiKey={apiKey}
              deepseekApiKey={deepseekApiKey}
              scenarios={scenarios}
              evals={evals}
              onComplete={handleOptimizationComplete}
            />
          )}

          {currentStep === 4 && finalResult && (
            <ResultsStep
              result={finalResult}
              sessionId={sessionId!}
              onStartOver={handleStartOver}
            />
          )}
        </div>
      </div>
    </main>
  );
}
