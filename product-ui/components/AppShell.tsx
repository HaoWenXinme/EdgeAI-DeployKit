"use client";

import { useEffect } from "react";
import {
  Activity,
  Boxes,
  Cpu,
  Database,
  FileText,
  Gauge,
  Layers3,
  Radar,
  Settings2,
  TerminalSquare,
} from "lucide-react";
import type { HealthResponse } from "@/lib/types";
import { Badge } from "./ui";

export type WorkspacePanel =
  | "overview"
  | "models"
  | "pipeline"
  | "benchmark"
  | "board"
  | "infer-result"
  | "reports"
  | "runtime";

const nav = [
  ["overview", "Overview", Radar],
  ["models", "Models", Database],
  ["pipeline", "Pipeline", Layers3],
  ["benchmark", "Benchmark", Gauge],
  ["board", "Board", Cpu],
  ["infer-result", "Infer Result", FileText],
  ["reports", "Reports", FileText],
  ["runtime", "Runtime", TerminalSquare],
] as const;


function useSmoothWheelScroll() {
  useEffect(() => {
    const scrollerElement = document.querySelector(".main-stage");

    if (!(scrollerElement instanceof HTMLElement)) {
      return;
    }

    const scroller: HTMLElement = scrollerElement;

    let targetScroll = scroller.scrollTop;
    let frame: number | null = null;

    const WHEEL_SCALE = 0.68;
    const EASE = 0.12;

    function maxScroll() {
      return Math.max(0, scroller.scrollHeight - scroller.clientHeight);
    }

    function clamp(value: number) {
      return Math.max(0, Math.min(value, maxScroll()));
    }

    function animate() {
      const current = scroller.scrollTop;
      const diff = targetScroll - current;

      if (Math.abs(diff) < 0.5) {
        scroller.scrollTop = targetScroll;
        frame = null;
        return;
      }

      scroller.scrollTop = current + diff * EASE;
      frame = window.requestAnimationFrame(animate);
    }

    function onWheel(event: WheelEvent) {
      if (event.ctrlKey) return;

      let delta = event.deltaY;

      if (event.deltaMode === 1) {
        delta *= 16;
      } else if (event.deltaMode === 2) {
        delta *= scroller.clientHeight;
      }

      /*
       * 触控板 delta 通常较小，本身已经连续，不强行接管。
       * 鼠标滚轮 delta 通常较大，所以拆成多帧滚动。
       */
      if (Math.abs(delta) < 40) {
        targetScroll = scroller.scrollTop;
        return;
      }

      event.preventDefault();

      targetScroll = clamp(targetScroll + delta * WHEEL_SCALE);

      if (frame === null) {
        frame = window.requestAnimationFrame(animate);
      }
    }

    function syncTarget() {
      if (frame === null) {
        targetScroll = scroller.scrollTop;
      }
    }

    scroller.addEventListener("wheel", onWheel, { passive: false });
    scroller.addEventListener("mousedown", syncTarget, { passive: true });
    scroller.addEventListener("touchstart", syncTarget, { passive: true });

    return () => {
      scroller.removeEventListener("wheel", onWheel);
      scroller.removeEventListener("mousedown", syncTarget);
      scroller.removeEventListener("touchstart", syncTarget);

      if (frame !== null) {
        window.cancelAnimationFrame(frame);
      }
    };
  }, []);
}


export function AppShell({
  children,
  health,
  activePanel = "overview",
  onPanelChange,
}: {
  children: React.ReactNode;
  health: HealthResponse;
  activePanel?: WorkspacePanel;
  onPanelChange?: (panel: WorkspacePanel) => void;
}) {
  useSmoothWheelScroll();

  const requiredChecks = health.checks.filter((item) => item.required !== false);
  const scoreChecks = requiredChecks.length ? requiredChecks : health.checks;
  const available = scoreChecks.filter((item) => item.available).length;
  const total = Math.max(scoreChecks.length, 1);
  const ratio = Math.round((available / total) * 100);

  return (
    <main className="layout-shell">
      <div className="workspace-image-backdrop" aria-hidden="true" />

      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-logo">
            <Boxes className="h-5 w-5" />
          </div>
          <div>
            <div className="sidebar-title">EdgeAI DeployKit</div>
            <div className="sidebar-subtitle">Deployment Workspace</div>
          </div>
        </div>

        <div className="sidebar-health">
          <div className="flex items-center justify-between">
            <span>Runtime health</span>
            <Badge tone={ratio >= 75 ? "green" : ratio >= 50 ? "amber" : "red"}>
              {available}/{total}
            </Badge>
          </div>
          <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/[0.08]">
            <div className="h-full rounded-full bg-[#ffd6e8]" style={{ width: `${ratio}%` }} />
          </div>
        </div>

        <nav className="mt-7 space-y-2">
          {nav.map(([id, label, Icon]) => {
            const active = activePanel === id;
            return (
              <button
                key={id}
                type="button"
                onClick={() => onPanelChange?.(id)}
                className={`nav-item ${active ? "nav-active" : ""}`}
              >
                <Icon className="h-4 w-4" />
                <span>{label}</span>
              </button>
            );
          })}
        </nav>

        <div className="absolute bottom-5 left-4 right-4 space-y-3">
          <div className="sidebar-context-card">
            <div className="flex items-center gap-2 text-sm text-white/70">
              <Activity className="h-4 w-4 text-[#ffd6e8]" />
              Runtime context
            </div>
            <div className="mt-4 space-y-3 text-sm leading-6 text-white/62">
              <div className="flex items-center justify-between">
                <span>Backend</span>
                <Badge tone="green">8001</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span>Workspace</span>
                <span className="font-mono text-white/45">local</span>
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={() => onPanelChange?.("runtime")}
            className="nav-item"
          >
            <Settings2 className="h-4 w-4" />
            <span>Settings & logs</span>
          </button>
        </div>
      </aside>

      <section className="main-stage">{children}</section>
    </main>
  );
}
