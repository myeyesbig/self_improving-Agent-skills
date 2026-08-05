"use client";

// =============================================================================
// 【文件头】UploadStep.tsx —— 第一步：上传技能 + 输入 API Key + 触发分析
// 职责：接收 .zip 或文件夹上传，把文件发给后端换取 session_id；收集用户在
//       组件内存里输入的 DashScope API Key；点击"Analyze"后调用 /api/analyze。
// 接收：父组件传入的 onComplete 回调（用于把 sessionId / apiKey 等上报）。
// 输出：调用 onComplete(...)，把数据交回父组件 page.tsx。
// 建议先看：uploadZip / uploadMultipleFiles（上传）→ handleAnalyze（分析）。
// 【初学者提示】API Key 只保存在这个组件的 useState 里（组件内存），
//       不写入 localStorage、不进 URL，仅在请求体中发送给后端。
// =============================================================================

import { useState, useRef, useEffect } from "react";
import { Upload, FileText, Loader2, Sparkles, FolderOpen } from "lucide-react";
import {
  uploadZip as apiUploadZip,
  uploadMultipleFiles as apiUploadMultipleFiles,
  analyzeSkill,
  loadExample,
  listExamples,
  ExampleSkill,
} from "@/lib/api";

interface UploadStepProps {
  onComplete: (
    sessionId: string,
    apiKey: string,
    metadata: any,
    scenarios: any[],
    evals: any[]
  ) => void;
}

export default function UploadStep({ onComplete }: UploadStepProps) {
  // 【初学者提示】isDragging 控制拖拽高亮；apiKey 是组件内存里的密钥；
  // isUploading / isAnalyzing 是按钮的加载态；sessionId 记录上传结果。
  const [isDragging, setIsDragging] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [fileList, setFileList] = useState<string[]>([]);
  const [metadata, setMetadata] = useState<any>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [exampleSkills, setExampleSkills] = useState<ExampleSkill[]>([]);
  const folderInputRef = useRef<HTMLInputElement>(null);

  // 挂载时从后端动态拉取示例技能列表（不再硬编码）。
  useEffect(() => {
    listExamples()
      .then(setExampleSkills)
      .catch(() => setExampleSkills([]));
  }, []);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    // 拖入的文件：优先找 .zip，找不到则当作多文件（文件夹）处理。
    const files = Array.from(e.dataTransfer.files);
    const zipFile = files.find((f) => f.name.endsWith(".zip"));
    if (zipFile) {
      await uploadZip(zipFile);
    } else if (files.length > 0) {
      await uploadMultipleFiles(files);
    }
  };

  const handleZipSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      await uploadZip(files[0]);
    }
  };

  const handleFolderSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      await uploadMultipleFiles(Array.from(files));
    }
  };

  // 【主流程】ZIP 上传：把文件交给统一 API 客户端（POST /api/upload），
  // 成功后把 session_id / 文件清单 / 元数据存入本组件 state。
  const uploadZip = async (file: File) => {
    setIsUploading(true);
    try {
      const data = await apiUploadZip(file);
      setSessionId(data.session_id);
      setFileList(data.file_list);
      setMetadata(data.metadata);
    } catch (error: any) {
      alert(error.message || "Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  // 【主流程】文件夹上传：走 /api/upload-files，用 webkitRelativePath 保留
  // 相对目录结构，其余逻辑与 ZIP 上传一致。
  const uploadMultipleFiles = async (files: File[]) => {
    setIsUploading(true);
    try {
      const data = await apiUploadMultipleFiles(files);
      setSessionId(data.session_id);
      setFileList(data.file_list);
      setMetadata(data.metadata);
    } catch (error: any) {
      alert(error.message || "Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  // 【主流程】分析请求：把 session_id 与 API Key 一起 POST 给 /api/analyze，
  // 后端返回模型生成的 scenarios（测试场景）和 evals（评估标准），
  // 最后调用 onComplete 把全部上下文上交给父组件，由父组件跳到步骤 2。
  const handleAnalyze = async () => {
    if (!apiKey || !sessionId) return;
    setIsAnalyzing(true);
    try {
      const data = await analyzeSkill(sessionId, apiKey);
      onComplete(sessionId, apiKey, metadata, data.scenarios, data.evals);
    } catch (error: any) {
      alert(error.message || "Analysis failed. Check your API key.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 示例技能：先 /api/examples/{path}/load 加载，再自动走一次分析，
  // 效果等同于"上传 + 分析"两步合一。
  const handleExampleSelect = async (examplePath: string) => {
    if (!apiKey) {
      alert("Please enter your DashScope API key first.");
      return;
    }
    setIsUploading(true);
    try {
      const loadData = await loadExample(examplePath);
      setSessionId(loadData.session_id);
      setFileList(loadData.file_list);
      setMetadata(loadData.metadata);
      setIsUploading(false);
      setIsAnalyzing(true);

      const analyzeData = await analyzeSkill(loadData.session_id, apiKey);
      onComplete(loadData.session_id, apiKey, loadData.metadata, analyzeData.scenarios, analyzeData.evals);
    } catch (error: any) {
      alert(error.message || "Failed to load example skill.");
      setSessionId(null);
    } finally {
      setIsUploading(false);
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {!sessionId ? (
        <>
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`glass rounded-2xl p-16 text-center transition-all ${
              isDragging ? "border-cyan-500 bg-cyan-500/10 scale-105" : "border-zinc-800 hover:border-zinc-700"
            }`}
          >
            <input type="file" accept=".zip" onChange={handleZipSelect} className="hidden" id="zip-upload" />
            <input
              type="file"
              ref={folderInputRef}
              onChange={handleFolderSelect}
              className="hidden"
              id="folder-upload"
              {...({ webkitdirectory: "", directory: "" } as any)}
              multiple
            />

            <div className="mb-6">
              {isUploading ? (
                <Loader2 className="w-16 h-16 mx-auto text-cyan-500 animate-spin" />
              ) : (
                <Upload className="w-16 h-16 mx-auto text-zinc-500" />
              )}
            </div>

            <h3 className="text-2xl font-semibold mb-2">
              {isUploading ? "Uploading..." : "Drop your skill here"}
            </h3>
            <p className="text-zinc-400 mb-4">Drag a .zip file or use the buttons below</p>

            <div className="flex justify-center gap-4">
              <label
                htmlFor="zip-upload"
                className="px-6 py-3 bg-zinc-800 hover:bg-zinc-700 rounded-xl cursor-pointer transition-colors flex items-center gap-2 text-sm font-medium"
              >
                <Upload className="w-4 h-4" />
                Upload .zip
              </label>
              <button
                onClick={() => folderInputRef.current?.click()}
                className="px-6 py-3 bg-zinc-800 hover:bg-zinc-700 rounded-xl cursor-pointer transition-colors flex items-center gap-2 text-sm font-medium"
              >
                <FolderOpen className="w-4 h-4" />
                Upload Folder
              </button>
            </div>
          </div>

          <div className="glass rounded-2xl p-6">
            <label className="block">
              <span className="text-sm font-medium text-zinc-400 mb-2 block">DashScope API Key</span>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter your DashScope API key"
                className="w-full px-4 py-3 bg-zinc-900 border border-zinc-800 rounded-lg focus:outline-none focus:border-cyan-500 transition-colors"
              />
              <span className="text-xs text-zinc-500 mt-1 block">
                Required for analysis. Stored locally, sent only to the backend.
              </span>
            </label>
          </div>

          <div>
            <p className="text-center text-zinc-500 mb-4">
              {isAnalyzing ? "Analyzing with Qwen-Agent..." : "Or try an example skill:"}
            </p>
            {exampleSkills.length === 0 ? (
              <p className="text-center text-sm text-zinc-600">
                No example skills available — upload a skill or try a .zip
              </p>
            ) : (
            <div className="grid grid-cols-2 gap-4">
              {exampleSkills.map((skill) => (
                <button
                  key={skill.path}
                  onClick={() => handleExampleSelect(skill.path)}
                  disabled={isUploading || isAnalyzing || !apiKey}
                  className={`glass rounded-xl p-6 text-left transition-all ${
                    apiKey && !isUploading && !isAnalyzing
                      ? "hover:border-cyan-500 hover:scale-105 cursor-pointer"
                      : "opacity-50 cursor-not-allowed"
                  }`}
                >
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    {skill.name}
                    {isAnalyzing && <Loader2 className="w-4 h-4 animate-spin text-cyan-500" />}
                  </h4>
                  <p className="text-sm text-zinc-400">{skill.description}</p>
                </button>
              ))}
            </div>
            )}
          </div>
        </>
      ) : (
        <div className="space-y-6">
          <div className="glass rounded-2xl p-8">
            <div className="flex items-start gap-4 mb-6">
              <FileText className="w-8 h-8 text-cyan-500" />
              <div className="flex-1">
                <h3 className="text-xl font-semibold mb-1">{metadata?.name || "Skill Uploaded"}</h3>
                {metadata?.description && <p className="text-zinc-400">{metadata.description}</p>}
              </div>
            </div>
            <div className="border-t border-zinc-800 pt-4 mt-4">
              <p className="text-sm text-zinc-500 mb-2">Files in skill:</p>
              <div className="flex flex-wrap gap-2">
                {fileList.map((file) => (
                  <span key={file} className="px-3 py-1 bg-zinc-800 rounded-full text-xs text-zinc-300">
                    {file}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="glass rounded-2xl p-8">
            <label className="block mb-4">
              <span className="text-sm font-medium text-zinc-400 mb-2 block">DashScope API Key</span>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter your DashScope API key"
                className="w-full px-4 py-3 bg-zinc-900 border border-zinc-800 rounded-lg focus:outline-none focus:border-cyan-500 transition-colors"
              />
            </label>
            <button
              onClick={handleAnalyze}
              disabled={!apiKey || isAnalyzing}
              className={`w-full py-4 rounded-xl font-semibold transition-all flex items-center justify-center gap-2 ${
                apiKey && !isAnalyzing ? "gradient-bg hover:scale-105" : "bg-zinc-800 text-zinc-500 cursor-not-allowed"
              }`}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing Skill...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Analyze Skill
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
