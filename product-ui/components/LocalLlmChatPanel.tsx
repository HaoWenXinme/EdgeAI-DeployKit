"use client";

import * as React from "react";
import { runLocalInferenceFlow } from "@/lib/api";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type LocalLlmChatPanelProps = {
  packageName?: string;
  onRefresh?: () => void | Promise<void>;
  onOpenReports?: () => void;
};

function buildPrompt(messages: ChatMessage[], nextUserMessage: string) {
  if (messages.length === 0) return `Please answer briefly: ${nextUserMessage.trim()}`;
  const turns = messages
    .slice(-6)
    .map((item) => `${item.role === "user" ? "User said" : "Assistant replied"}: ${item.content.trim()}`)
    .join("\n");
  return `Previous conversation:\n${turns}\nCurrent user message: ${nextUserMessage.trim()}\nReply briefly:`;
}

function extractResponse(payload: unknown) {
  const data = payload as {
    task_result?: { conversation?: { response?: string }; summary?: { primary?: string } };
    stages?: Array<{ stage?: string; output?: string }>;
  };
  const direct = data.task_result?.conversation?.response;
  if (direct && direct.trim()) return direct.trim();
  const primary = data.task_result?.summary?.primary;
  if (primary && primary.trim() && primary !== "Chat output generated.") return primary.trim();
  const runStage = data.stages?.find((stage) => stage.stage === "local-run");
  if (!runStage?.output) return "";
  try {
    const parsed = JSON.parse(runStage.output);
    return String(parsed.response || parsed.outputs?.[0]?.text || "").trim();
  } catch {
    return runStage.output.trim();
  }
}

export function LocalLlmChatPanel({ packageName, onRefresh, onOpenReports }: LocalLlmChatPanelProps) {
  const [messages, setMessages] = React.useState<ChatMessage[]>([]);
  const [input, setInput] = React.useState("Hello, introduce yourself briefly.");
  const [maxTokens, setMaxTokens] = React.useState(64);
  const [temperature, setTemperature] = React.useState(0.2);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");

  const validPackage = Boolean(packageName && packageName !== "model" && packageName !== "<package>");

  async function sendMessage() {
    const text = input.trim();
    if (!text || !validPackage || busy) return;
    const prompt = buildPrompt(messages, text);
    const optimistic = [...messages, { role: "user" as const, content: text }];
    setMessages(optimistic);
    setInput("");
    setError("");
    setBusy(true);
    try {
      const result = await runLocalInferenceFlow({
        package_name: packageName || "",
        prompt,
        max_tokens: maxTokens,
        temperature,
        force_report: true,
      });
      const response = extractResponse(result) || "The local model did not generate text for this prompt.";
      setMessages([...optimistic, { role: "assistant", content: response }]);
      await onRefresh?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setMessages(messages);
      setInput(text);
    } finally {
      setBusy(false);
    }
  }

  function handleKeyDown(event: any) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  return (
    <section className="rounded-[30px] border border-emerald-300/20 bg-slate-950/55 p-6 shadow-2xl shadow-black/25">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="text-xs font-black uppercase tracking-[0.25em] text-emerald-100/80">Local GGUF Chat</div>
          <h2 className="mt-2 text-2xl font-black text-white">本地大语言模型对话</h2>
          <p className="mt-2 max-w-3xl text-sm leading-7 text-slate-300">
            当前 package 会作为本地 GGUF 部署运行，消息直接发送到本机 llama.cpp runtime，不调用云端模型。
          </p>
        </div>
        <div className="rounded-2xl border border-white/10 bg-black/25 px-4 py-3 text-xs text-slate-300">
          <div className="text-slate-500">Package</div>
          <div className="mt-1 font-mono font-bold text-emerald-100">{packageName || "waiting"}</div>
        </div>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-[1fr_220px]">
        <div className="min-h-[320px] rounded-2xl border border-white/10 bg-black/25 p-4">
          <div className="flex h-[300px] flex-col gap-3 overflow-auto pr-1">
            {messages.length === 0 ? (
              <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-white/10 text-sm text-slate-500">
                等待第一条对话
              </div>
            ) : (
              messages.map((message, index) => {
                const isUser = message.role === "user";
                return (
                  <div key={`${message.role}-${index}`} className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
                    {!isUser ? <div className="mt-1 h-6 w-6 shrink-0 rounded-full border border-emerald-300/25 bg-emerald-300/10 text-center text-[10px] font-black leading-6 text-emerald-100">AI</div> : null}
                    <div className={`max-w-[82%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-6 ${isUser ? "bg-cyan-300/15 text-cyan-50" : "bg-white/[0.06] text-slate-100"}`}>
                      {message.content}
                    </div>
                    {isUser ? <div className="mt-1 h-6 w-6 shrink-0 rounded-full border border-cyan-300/25 bg-cyan-300/10 text-center text-[10px] font-black leading-6 text-cyan-100">ME</div> : null}
                  </div>
                );
              })
            )}
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-[1fr_auto]">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              disabled={!validPackage || busy}
              className="min-h-[88px] resize-none rounded-2xl border border-white/10 bg-slate-950/80 px-4 py-3 text-sm leading-6 text-white outline-none transition focus:border-emerald-300/40 disabled:opacity-50"
            />
            <button
              type="button"
              onClick={() => void sendMessage()}
              disabled={!validPackage || busy || !input.trim()}
              className="flex min-h-[88px] items-center justify-center gap-2 rounded-2xl border border-emerald-300/30 bg-emerald-300/15 px-5 text-sm font-black text-emerald-100 transition hover:bg-emerald-300/20 disabled:cursor-not-allowed disabled:opacity-45"
            >
              {busy ? "Running..." : "Send"}
            </button>
          </div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
          <label className="block text-xs font-bold uppercase tracking-[0.18em] text-slate-400">
            Max tokens
            <input
              type="number"
              min={16}
              max={1024}
              value={maxTokens}
              onChange={(event) => setMaxTokens(Number(event.target.value || 128))}
              className="mt-2 w-full rounded-xl border border-white/10 bg-slate-950/70 px-3 py-2 font-mono text-sm text-white"
            />
          </label>
          <label className="mt-4 block text-xs font-bold uppercase tracking-[0.18em] text-slate-400">
            Temperature
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(event) => setTemperature(Number(event.target.value || 0.7))}
              className="mt-2 w-full rounded-xl border border-white/10 bg-slate-950/70 px-3 py-2 font-mono text-sm text-white"
            />
          </label>
          <button
            type="button"
            onClick={() => setMessages([])}
            className="mt-4 w-full rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 text-xs font-bold text-slate-200 hover:bg-white/[0.08]"
          >
            Clear chat
          </button>
          <button
            type="button"
            onClick={onOpenReports}
            className="mt-2 w-full rounded-xl border border-cyan-300/25 bg-cyan-300/10 px-3 py-2 text-xs font-bold text-cyan-100 hover:bg-cyan-300/20"
          >
            Open report
          </button>
        </div>
      </div>

      {error ? (
        <div className="mt-4 rounded-xl border border-rose-300/20 bg-rose-500/10 p-3 text-xs leading-6 text-rose-100">
          {error}
        </div>
      ) : null}
    </section>
  );
}
