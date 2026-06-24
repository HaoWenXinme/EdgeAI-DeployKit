type CheckItem = {
  name: string;
  path: string;
  ok: boolean;
  required: boolean;
  category: string;
};

type HealthLike = {
  score?: string;
  passed?: number;
  total?: number;
  checks?: unknown;
};

const fallbackChecks: CheckItem[] = [
  { name: "edgeai CLI", path: "/root/edge-ai-deploy-kit/.venv/bin/edgeai", ok: true, required: true, category: "core" },
  { name: "Python", path: "/root/edge-ai-deploy-kit/.venv/bin/python3", ok: true, required: true, category: "core" },
  { name: "cmake", path: "/usr/bin/cmake", ok: true, required: false, category: "native-build" },
  { name: "make", path: "/usr/bin/make", ok: true, required: false, category: "native-build" },
  { name: "gcc", path: "/usr/bin/gcc", ok: true, required: false, category: "native-build" },
  { name: "g++", path: "/usr/bin/g++", ok: true, required: false, category: "native-build" },
  { name: "qemu-system-aarch64", path: "/usr/local/bin/qemu-system-aarch64", ok: true, required: false, category: "board" },
  { name: "atc", path: "atc", ok: false, required: false, category: "board" },
  { name: "docker", path: "/usr/bin/docker", ok: true, required: false, category: "board" },
  {
    name: "openEuler aarch64 SDK",
    path: "/opt/openeuler-aarch64/environment-setup-aarch64-openeuler-linux",
    ok: false,
    required: false,
    category: "board",
  },
];

function valueToCheck(name: string, value: unknown): CheckItem {
  if (typeof value === "boolean") {
    return { name, path: name, ok: value, required: true, category: "core" };
  }

  if (typeof value === "string") {
    return { name, path: value, ok: Boolean(value), required: true, category: "core" };
  }

  if (value && typeof value === "object") {
    const item = value as {
      name?: string;
      label?: string;
      path?: string;
      value?: string;
      command?: string;
      ok?: boolean;
      available?: boolean;
      status?: string;
      required?: boolean;
      category?: string;
    };

    const statusText = String(item.status || "").toLowerCase();
    const ok =
      typeof item.ok === "boolean"
        ? item.ok
        : typeof item.available === "boolean"
          ? item.available
          : statusText === "ok" || statusText === "ready" || statusText === "success";

    return {
      name: item.name || item.label || name,
      path: item.path || item.value || item.command || name,
      ok,
      required: item.required !== false,
      category: item.category || "core",
    };
  }

  return { name, path: name, ok: false, required: true, category: "core" };
}

function normalizeChecks(health?: HealthLike): CheckItem[] {
  const checks = health?.checks;

  if (Array.isArray(checks)) {
    return checks.map((item, index) => valueToCheck(`tool-${index + 1}`, item));
  }

  if (checks && typeof checks === "object") {
    return Object.entries(checks as Record<string, unknown>).map(([name, value]) =>
      valueToCheck(name, value),
    );
  }

  return fallbackChecks;
}

function scoreText(health?: HealthLike, checks?: CheckItem[]) {
  if (health?.score) return health.score;

  if (typeof health?.passed === "number" && typeof health?.total === "number") {
    return `${health.passed}/${health.total}`;
  }

  if (checks?.length) {
    const required = checks.filter((item) => item.required);
    const scoreChecks = required.length ? required : checks;
    const passed = scoreChecks.filter((item) => item.ok).length;
    return `${passed}/${scoreChecks.length}`;
  }

  return "0/0";
}

export function RuntimeChecksPanel({ health }: { health?: HealthLike }) {
  const checks = normalizeChecks(health);
  const score = scoreText(health, checks);
  const requiredChecks = checks.filter((item) => item.required);
  const optionalChecks = checks.filter((item) => !item.required);

  return (
    <section className="runtime-assets-panel">
      <div className="runtime-assets-head">
        <div>
          <div className="product-kicker">Health</div>
          <h2>Runtime capability</h2>
        </div>

        <strong>{score}</strong>
      </div>

      <div className="runtime-assets-table">
        <div className="runtime-assets-row runtime-assets-row-head">
          <span>Tool</span>
          <span>Path</span>
          <span>Status</span>
        </div>

        {requiredChecks.map((item) => (
          <div key={`${item.name}-${item.path}`} className="runtime-assets-row">
            <span>{item.name}</span>
            <code>{item.path}</code>
            <em className={item.ok ? "runtime-status-ok" : "runtime-status-missing"}>
              {item.ok ? "Ready" : "Missing"}
            </em>
          </div>
        ))}

        {optionalChecks.map((item) => (
          <div key={`${item.name}-${item.path}`} className="runtime-assets-row">
            <span>{item.name}</span>
            <code>{item.path}</code>
            <em className={item.ok ? "runtime-status-ok" : "runtime-status-missing"}>
              {item.ok ? "Ready" : `Optional / ${item.category}`}
            </em>
          </div>
        ))}
      </div>
    </section>
  );
}
