"use client";

// =============================================================================
// 【文件头】StepIndicator.tsx —— 顶部四步进度指示器
// 职责：根据当前步骤号渲染"上传 → 配置 → 优化 → 结果"的步骤条。
// 接收：父组件传入的 currentStep（1-4）。
// 输出：纯展示组件，不修改任何状态。
// 建议先看：下方对 currentStep 的三段条件判断（已完成 / 当前 / 未到）。
// 【初学者提示】核心是条件渲染：已完成的步骤显示对勾并点亮，当前步骤放大
//       高亮，未到步骤显示灰色数字；步骤之间的连接线按完成情况着色。
// =============================================================================

import { Check } from "lucide-react";

interface StepIndicatorProps {
  currentStep: number;
}

const steps = [
  { number: 1, name: "Upload" },
  { number: 2, name: "Configure" },
  { number: 3, name: "Optimize" },
  { number: 4, name: "Results" },
];

export default function StepIndicator({ currentStep }: StepIndicatorProps) {
  return (
    <div className="flex items-center justify-center gap-4">
      {steps.map((step, index) => (
        <div key={step.number} className="flex items-center">
          <div className="flex items-center gap-3">
            <div
              className={`
                w-12 h-12 rounded-full flex items-center justify-center font-semibold
                transition-all duration-300
                ${
                  currentStep > step.number
                    ? "gradient-bg text-white"
                    : currentStep === step.number
                    ? "gradient-bg text-white scale-110"
                    : "bg-zinc-800 text-zinc-500"
                }
              `}
            >
              {/* 条件渲染：已完成的步骤显示对勾，否则显示步骤数字。 */}
              {currentStep > step.number ? (
                <Check className="w-6 h-6" />
              ) : (
                step.number
              )}
            </div>
            <span
              className={`
                text-sm font-medium transition-colors
                ${
                  currentStep >= step.number
                    ? "text-white"
                    : "text-zinc-500"
                }
              `}
            >
              {step.name}
            </span>
          </div>

          {index < steps.length - 1 && (
            <div
              className={`
                w-16 h-0.5 mx-4 transition-colors
                ${
                  currentStep > step.number
                    ? "bg-gradient-to-r from-cyan-500 to-teal-500"
                    : "bg-zinc-800"
                }
              `}
            />
          )}
        </div>
      ))}
    </div>
  );
}
