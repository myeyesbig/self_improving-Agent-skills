"use client";

// =============================================================================
// 【文件头】ConfigStep.tsx —— 第二步：配置测试场景与评估标准
// 职责：展示并编辑上一步生成的 scenarios（测试场景）和 evals（评估标准），
//       支持勾选 / 增删 / 编辑 / 重新生成，最后保存到后端并进入优化。
// 接收：父组件传入的 sessionId / apiKey / scenarios / evals 及变更回调。
// 输出：把勾选后的结果上报给父组件（onScenariosChange / onEvalsChange），
//       并调用 onComplete() 推动步骤前进。
// 建议先看：handleContinue（保存 + 前进）与 handleRegenerate（重新生成）。
// 【初学者提示】组件把 props 传入的数据复制到局部 state 并附加 selected /
//       editing 标记；所有更新都用展开运算符（...）做"不可变更新"，再配合
//       map / filter 生成新数组，React 才能正确感知变化并重渲染。
// =============================================================================

import { useState } from "react";
import { CheckSquare, Square, Plus, X, RefreshCw, ArrowRight } from "lucide-react";
import { regenerateConfig, updateConfig } from "@/lib/api";

interface ConfigStepProps {
  sessionId: string;
  apiKey: string;
  scenarios: any[];
  evals: any[];
  onScenariosChange: (scenarios: any[]) => void;
  onEvalsChange: (evals: any[]) => void;
  onComplete: () => void;
}

export default function ConfigStep({
  sessionId,
  apiKey,
  scenarios: initialScenarios,
  evals: initialEvals,
  onScenariosChange,
  onEvalsChange,
  onComplete,
}: ConfigStepProps) {
  // 【初学者提示】props 里的是父组件存的"初始数据"；这里用 map 给每一项
  // 附加 UI 状态：selected（是否勾选参与优化）、editing（是否处于编辑态）。
  const [scenarios, setScenarios] = useState(
    initialScenarios.map((s) => ({ ...s, selected: true, editing: false }))
  );
  const [evals, setEvals] = useState(
    initialEvals.map((e) => ({ ...e, selected: true, editing: false }))
  );
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [showAddScenario, setShowAddScenario] = useState(false);
  const [showAddEval, setShowAddEval] = useState(false);

  // 【初学者提示】以下五个 handler 都遵循同一模式：用 map / filter 生成新
  // 数组（不改原数组），只有 id 匹配的那一项被替换/删除 —— 这就是不可变更新。
  const handleScenarioToggle = (id: number) => {
    setScenarios((prev) =>
      prev.map((s) => (s.id === id ? { ...s, selected: !s.selected } : s))
    );
  };

  const handleEvalToggle = (id: number) => {
    setEvals((prev) =>
      prev.map((e) => (e.id === id ? { ...e, selected: !e.selected } : e))
    );
  };

  // 编辑任意字段：用 [field] 动态键展开覆盖，其他字段保持不变。
  const handleScenarioEdit = (id: number, field: string, value: string) => {
    setScenarios((prev) =>
      prev.map((s) => (s.id === id ? { ...s, [field]: value } : s))
    );
  };

  const handleEvalEdit = (id: number, field: string, value: string) => {
    setEvals((prev) =>
      prev.map((e) => (e.id === id ? { ...e, [field]: value } : e))
    );
  };

  const handleDeleteScenario = (id: number) => {
    setScenarios((prev) => prev.filter((s) => s.id !== id));
  };

  const handleDeleteEval = (id: number) => {
    setEvals((prev) => prev.filter((e) => e.id !== id));
  };

  // 新增：id 取现有最大值 +1，保证不重复；新项默认选中并进入编辑态。
  const handleAddScenario = () => {
    const newId = Math.max(...scenarios.map((s) => s.id), 0) + 1;
    setScenarios((prev) => [
      ...prev,
      {
        id: newId,
        description: "New scenario",
        input: "",
        selected: true,
        editing: true,
      },
    ]);
    setShowAddScenario(false);
  };

  const handleAddEval = () => {
    const newId = Math.max(...evals.map((e) => e.id), 0) + 1;
    setEvals((prev) => [
      ...prev,
      {
        id: newId,
        name: "New criterion",
        question: "",
        pass_condition: "",
        fail_condition: "",
        selected: true,
        editing: true,
      },
    ]);
    setShowAddEval(false);
  };

  // 【主流程】重新生成：让后端用 /api/regenerate 重新调用模型生成一份新的
  // scenarios / evals，然后覆盖本地 state（新项默认全部选中）。
  const handleRegenerate = async () => {
    setIsRegenerating(true);

    try {
      const data = await regenerateConfig(sessionId, apiKey);
      setScenarios(data.scenarios.map((s: any) => ({ ...s, selected: true })));
      setEvals(data.evals.map((e: any) => ({ ...e, selected: true })));
    } catch (error) {
      alert("Failed to regenerate. Please try again.");
    } finally {
      setIsRegenerating(false);
    }
  };

  // 【主流程】继续：筛选出勾选项，先保存到后端（/api/update-config），再把
  // 勾选结果上报给父组件（page.tsx 的 state），最后调用 onComplete 跳到步骤 3。
  const handleContinue = async () => {
    const selectedScenarios = scenarios.filter((s) => s.selected);
    const selectedEvals = evals.filter((e) => e.selected);

    if (selectedScenarios.length === 0 || selectedEvals.length === 0) {
      alert("Please select at least one scenario and one evaluation criterion");
      return;
    }

    try {
      await updateConfig(sessionId, selectedScenarios, selectedEvals);
      onScenariosChange(selectedScenarios);
      onEvalsChange(selectedEvals);
      onComplete();
    } catch (error) {
      alert("Failed to save configuration. Please try again.");
    }
  };

  // 派生计数：用于按钮可用性与"已选数量"角标（不改 state，直接由渲染读取）。
  const selectedScenarioCount = scenarios.filter((s) => s.selected).length;
  const selectedEvalCount = evals.filter((e) => e.selected).length;

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <div className="glass rounded-2xl p-8">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold">Test Scenarios</h2>
            <span className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded-full text-sm font-medium">
              {selectedScenarioCount} selected
            </span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setShowAddScenario(true)}
              className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Add
            </button>
            <button
              onClick={handleRegenerate}
              disabled={isRegenerating}
              className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors flex items-center gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${isRegenerating ? "animate-spin" : ""}`} />
              Regenerate
            </button>
          </div>
        </div>

        <div className="space-y-4">
          {showAddScenario && (
            <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
              <button
                onClick={handleAddScenario}
                className="w-full py-3 border-2 border-dashed border-zinc-700 rounded-lg hover:border-cyan-500 transition-colors text-zinc-400 hover:text-white"
              >
                Click to add scenario
              </button>
            </div>
          )}

          {scenarios.map((scenario) => (
            <div
              key={scenario.id}
              className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg hover:border-zinc-700 transition-colors"
            >
              <div className="flex items-start gap-3">
                <button
                  onClick={() => handleScenarioToggle(scenario.id)}
                  className="mt-1 text-cyan-500 hover:text-cyan-400"
                >
                  {scenario.selected ? (
                    <CheckSquare className="w-5 h-5" />
                  ) : (
                    <Square className="w-5 h-5" />
                  )}
                </button>

                <div className="flex-1 space-y-2">
                  <input
                    type="text"
                    value={scenario.description}
                    onChange={(e) =>
                      handleScenarioEdit(scenario.id, "description", e.target.value)
                    }
                    className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg focus:outline-none focus:border-cyan-500"
                  />
                  <textarea
                    value={scenario.input}
                    onChange={(e) =>
                      handleScenarioEdit(scenario.id, "input", e.target.value)
                    }
                    rows={3}
                    placeholder="Test input..."
                    className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg focus:outline-none focus:border-cyan-500 resize-none"
                  />
                </div>

                <button
                  onClick={() => handleDeleteScenario(scenario.id)}
                  className="mt-1 text-zinc-500 hover:text-red-400 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="glass rounded-2xl p-8">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold">Evaluation Criteria</h2>
            <span className="px-3 py-1 bg-cyan-500/20 text-cyan-400 rounded-full text-sm font-medium">
              {selectedEvalCount} selected
            </span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setShowAddEval(true)}
              className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Add
            </button>
          </div>
        </div>

        <div className="space-y-4">
          {showAddEval && (
            <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
              <button
                onClick={handleAddEval}
                className="w-full py-3 border-2 border-dashed border-zinc-700 rounded-lg hover:border-cyan-500 transition-colors text-zinc-400 hover:text-white"
              >
                Click to add criterion
              </button>
            </div>
          )}

          {evals.map((evalItem) => (
            <div
              key={evalItem.id}
              className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg hover:border-zinc-700 transition-colors"
            >
              <div className="flex items-start gap-3">
                <button
                  onClick={() => handleEvalToggle(evalItem.id)}
                  className="mt-1 text-cyan-500 hover:text-cyan-400"
                >
                  {evalItem.selected ? (
                    <CheckSquare className="w-5 h-5" />
                  ) : (
                    <Square className="w-5 h-5" />
                  )}
                </button>

                <div className="flex-1 space-y-2">
                  <input
                    type="text"
                    value={evalItem.name}
                    onChange={(e) =>
                      handleEvalEdit(evalItem.id, "name", e.target.value)
                    }
                    className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg focus:outline-none focus:border-cyan-500 font-semibold"
                  />
                  <input
                    type="text"
                    value={evalItem.question}
                    onChange={(e) =>
                      handleEvalEdit(evalItem.id, "question", e.target.value)
                    }
                    placeholder="Yes/no question..."
                    className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg focus:outline-none focus:border-cyan-500"
                  />
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="text"
                      value={evalItem.pass_condition}
                      onChange={(e) =>
                        handleEvalEdit(evalItem.id, "pass_condition", e.target.value)
                      }
                      placeholder="Pass condition..."
                      className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg focus:outline-none focus:border-green-500 text-sm"
                    />
                    <input
                      type="text"
                      value={evalItem.fail_condition}
                      onChange={(e) =>
                        handleEvalEdit(evalItem.id, "fail_condition", e.target.value)
                      }
                      placeholder="Fail condition..."
                      className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg focus:outline-none focus:border-red-500 text-sm"
                    />
                  </div>
                </div>

                <button
                  onClick={() => handleDeleteEval(evalItem.id)}
                  className="mt-1 text-zinc-500 hover:text-red-400 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      <button
        onClick={handleContinue}
        disabled={selectedScenarioCount === 0 || selectedEvalCount === 0}
        className={`
          w-full py-4 rounded-xl font-semibold transition-all flex items-center justify-center gap-2
          ${
            selectedScenarioCount > 0 && selectedEvalCount > 0
              ? "gradient-bg hover:scale-105"
              : "bg-zinc-800 text-zinc-500 cursor-not-allowed"
          }
        `}
      >
        Start Optimization
        <ArrowRight className="w-5 h-5" />
      </button>
    </div>
  );
}
