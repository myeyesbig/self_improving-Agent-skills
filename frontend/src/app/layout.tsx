import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import ThemeToggle from "@/components/ThemeToggle";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "SkillForge",
  description: "Self-forging agent skills with Qwen-Agent — a 3-agent loop that hammers raw SKILL.md into production-grade tools",
};

// =============================================================================
// 【文件头】layout.tsx —— Next.js 根布局
// 职责：包裹所有页面的 HTML 骨架，注入主题脚本、全局字体与 ThemeToggle。
// 接收：子页面内容（children）。
// 输出：渲染 <html>/<body> 结构，所有页面共享。
// 建议先看：<head> 里的内联脚本（主题首帧恢复）与 suppressHydrationWarning。
// 【初学者提示】hydration（水合）指服务端渲染的静态 HTML 与客户端 React 状态
//       合并的过程。这里在 <head> 放内联脚本，先在浏览器解析 HTML 时立即读
//       localStorage 设置主题 class，避免页面先闪一下浅色再变深色。
//       suppressHydrationWarning 是告诉 React：<html> 上的 class 在服务端与
//       客户端可能不一致，这是预期行为，不要报警告。
// =============================================================================

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{
          __html: `
            (function() {
              var theme = localStorage.getItem('theme');
              if (theme === 'light') {
                document.documentElement.classList.add('light');
                document.documentElement.classList.remove('dark');
              } else {
                document.documentElement.classList.add('dark');
                document.documentElement.classList.remove('light');
              }
            })();
          `
        }} />
      </head>
      <body className={inter.className}>
        <ThemeToggle />
        {children}
      </body>
    </html>
  );
}
