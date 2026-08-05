"use client";

// =============================================================================
// 【文件头】Logo.tsx —— SkillForge 品牌标识
// 职责：渲染内联 SVG 的 SkillForge 标志（砧台 + 锤击的青色渐变动画），
//       作为页头与品牌区共用的视觉锚点。
// 接收：可选 size（像素，默认 40）。
// 输出：一个带渐变的 SVG 图标。
// =============================================================================

interface LogoProps {
  size?: number;
}

export default function Logo({ size = 40 }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="SkillForge logo"
    >
      <defs>
        <linearGradient id="sf-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="100%" stopColor="#14b8a6" />
        </linearGradient>
      </defs>
      {/* Anvil */}
      <path d="M5 22 L10 13 L16 16 L22 13 L27 22 Z" fill="url(#sf-g)" />
      {/* Forge opening */}
      <rect x="13.5" y="16" width="5" height="4" rx="1" fill="#09090b" />
      {/* Base */}
      <rect x="14.5" y="19" width="3" height="7" rx="1.5" fill="url(#sf-g)" />
    </svg>
  );
}
