const FIELD_META = [
  { key: "age", label: "Age", type: "number", min: 18, max: 120, step: 1, def: 54 },
  { key: "sex", label: "Sex (1=male, 0=female)", type: "number", min: 0, max: 1, step: 1, def: 1 },
  { key: "cp", label: "Chest pain type (0–3)", type: "number", min: 0, max: 3, step: 1, def: 2 },
  { key: "trestbps", label: "Resting BP (mm Hg)", type: "number", min: 80, max: 250, step: 1, def: 140 },
  { key: "chol", label: "Cholesterol (mg/dl)", type: "number", min: 100, max: 600, step: 1, def: 239 },
  { key: "fbs", label: "Fasting BS > 120 (1=yes)", type: "number", min: 0, max: 1, step: 1, def: 0 },
  { key: "restecg", label: "Resting ECG (0–2)", type: "number", min: 0, max: 2, step: 1, def: 0 },
  { key: "thalach", label: "Max heart rate", type: "number", min: 60, max: 220, step: 1, def: 153 },
  { key: "exang", label: "Exercise angina (1=yes)", type: "number", min: 0, max: 1, step: 1, def: 0 },
  { key: "oldpeak", label: "ST depression (oldpeak)", type: "number", min: 0, max: 10, step: 0.1, def: 1.2 },
  { key: "slope", label: "ST slope (0–2)", type: "number", min: 0, max: 2, step: 1, def: 1 },
  { key: "ca", label: "Major vessels (0–3)", type: "number", min: 0, max: 3, step: 1, def: 0 },
  { key: "thal", label: "Thal / perfusion (0–9; UCI may use 3,6,7)", type: "number", min: 0, max: 9, step: 1, def: 2 },
];

/** Base URL for the FastAPI app (no trailing slash). */
function getApiBase() {
  const fromMeta = document.querySelector('meta[name="api-base"]')?.getAttribute("content")?.trim();
  if (fromMeta) return fromMeta.replace(/\/$/, "");

  const loc = window.location;
  if (loc.protocol === "file:") {
    return "http://127.0.0.1:8000";
  }

  const port = loc.port || (loc.protocol === "https:" ? "443" : loc.protocol === "http:" ? "80" : "");
  const devPorts = new Set(["5500", "5501", "5502", "3000", "5173", "4173", "4321", "8888"]);
  if (devPorts.has(String(port))) {
    const p = loc.protocol === "https:" ? "https" : "http";
    return `${p}://${loc.hostname}:8000`;
  }

  return `${loc.protocol}//${loc.host}`;
}

function apiUrl(path) {
  const base = getApiBase();
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${base}${p}`;
}

function formatErrorDetail(detail) {
  if (detail == null || detail === "") return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((e) => {
        if (e && typeof e === "object") {
          const loc = Array.isArray(e.loc) ? e.loc.join(".") : "";
          const msg = e.msg || e.message || "";
          return [loc, msg].filter(Boolean).join(": ");
        }
        return String(e);
      })
      .filter(Boolean)
      .join(" · ");
  }
  if (typeof detail === "object") {
    return detail.msg || detail.message || detail.error || JSON.stringify(detail);
  }
  return String(detail);
}

async function readJsonSafe(res) {
  const text = await res.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { _raw: text.slice(0, 280) };
  }
}

function bootUi() {
  document.body.classList.add("is-loaded");
  document.querySelectorAll("[data-reveal]").forEach((el, i) => {
    el.style.setProperty("--reveal-i", String(i));
    requestAnimationFrame(() => el.classList.add("is-visible"));
  });
}

function setApiPill(status, text) {
  const pill = document.getElementById("api-pill");
  if (!pill) return;
  pill.classList.remove("topbar__pill--ok", "topbar__pill--warn", "topbar__pill--bad");
  if (status === "ok") pill.classList.add("topbar__pill--ok");
  else if (status === "warn") pill.classList.add("topbar__pill--warn");
  else if (status === "bad") pill.classList.add("topbar__pill--bad");
  const t = pill.querySelector(".topbar__pill-text");
  if (t) t.textContent = text;
}

async function pingApi() {
  try {
    const res = await fetch(apiUrl("/api/health"), { method: "GET" });
    const body = await readJsonSafe(res);
    if (!res.ok) throw new Error("bad");
    const n = Array.isArray(body.models_loaded) ? body.models_loaded.length : 0;
    setApiPill(body.status === "ok" && n ? "ok" : "warn", n ? `API live · ${n} models` : "API up · no models");
  } catch {
    setApiPill("bad", "API unreachable");
  }
}

const fieldsRoot = document.getElementById("fields");

FIELD_META.forEach((f, i) => {
  const row = document.createElement("div");
  row.className = "row row--field";
  row.style.setProperty("--i", String(i));
  row.innerHTML = `
    <label for="${f.key}">${f.label}</label>
    <input id="${f.key}" name="${f.key}" type="${f.type}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.def}" required />
  `;
  fieldsRoot.appendChild(row);
});

bootUi();

async function loadMetrics() {
  const tbody = document.querySelector("#metrics-table tbody");
  tbody.innerHTML = "";
  try {
    const res = await fetch(apiUrl("/api/metrics"));
    if (!res.ok) throw new Error("metrics unavailable");
    const data = await res.json();
    const rows = Object.entries(data);
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="6">Train models first to populate metrics.</td></tr>`;
      return;
    }
    const label = (k) =>
      ({
        logistic_regression: "Logistic Regression",
        decision_tree: "Decision Tree",
        random_forest: "Random Forest",
        mlp_classifier: "Neural Network (MLP)",
      }[k] || k);
    rows.forEach(([name, m], r) => {
      const tr = document.createElement("tr");
      tr.className = "table-row";
      tr.style.setProperty("--r", String(r));
      tr.innerHTML = `
        <td>${label(name)}</td>
        <td>${m.accuracy?.toFixed?.(3) ?? "—"}</td>
        <td>${m.precision?.toFixed?.(3) ?? "—"}</td>
        <td>${m.recall?.toFixed?.(3) ?? "—"}</td>
        <td>${m.f1?.toFixed?.(3) ?? "—"}</td>
        <td>${m.roc_auc == null ? "—" : m.roc_auc.toFixed(3)}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch {
    tbody.innerHTML = `<tr><td colspan="6">Start the API and run training to see metrics.</td></tr>`;
  }
}

function renderFeatureImportance(data) {
  const tabs = document.getElementById("fi-tabs");
  const content = document.getElementById("fi-content");
  tabs.innerHTML = "";
  content.innerHTML = "";
  const keys = Object.keys(data || {});
  if (!keys.length) {
    content.textContent = "No feature importance artifacts found.";
    return;
  }
  let active = keys[0];
  const renderPanel = () => {
    content.innerHTML = "";
    const items = Object.entries(data[active] || {}).sort((a, b) => b[1] - a[1]);
    const max = Math.max(...items.map(([, v]) => v), 1e-9);
    items.forEach(([feat, val], f) => {
      const pct = (val / max) * 100;
      const row = document.createElement("div");
      row.className = "fi-row";
      row.style.setProperty("--f", String(f));
      row.style.setProperty("--target", `${pct}%`);
      row.innerHTML = `
        <div>
          <div class="fi-name">${feat}</div>
          <div class="bar"><span></span></div>
        </div>
        <div class="fi-val">${val.toFixed(4)}</div>
      `;
      content.appendChild(row);
    });
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        content.querySelectorAll(".fi-row").forEach((row) => row.classList.add("fi-row--in"));
      });
    });
  };
  keys.forEach((k, idx) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tab" + (idx === 0 ? " active" : "");
    b.textContent = k.replaceAll("_", " ");
    b.addEventListener("click", () => {
      active = k;
      [...tabs.querySelectorAll(".tab")].forEach((t) => t.classList.remove("active"));
      b.classList.add("active");
      renderPanel();
    });
    tabs.appendChild(b);
  });
  renderPanel();
}

async function loadFi() {
  try {
    const res = await fetch(apiUrl("/api/feature-importance"));
    if (!res.ok) throw new Error("no fi");
    renderFeatureImportance(await res.json());
  } catch {
    renderFeatureImportance({});
  }
}

document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = document.getElementById("form-error");
  err.textContent = "";
  const fd = new FormData(e.target);
  const patient = {};
  for (const f of FIELD_META) {
    const raw = fd.get(f.key);
    const num = f.key === "oldpeak" ? Number(raw) : Number.parseInt(String(raw), 10);
    if (!Number.isFinite(num)) {
      err.textContent = `Invalid number for “${f.label}”.`;
      return;
    }
    patient[f.key] = num;
  }
  const model = String(fd.get("model") || "random_forest");
  const btn = e.target.querySelector("button.primary");
  btn?.setAttribute("disabled", "true");
  try {
    const res = await fetch(apiUrl("/api/predict"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient, model }),
    });
    const body = await readJsonSafe(res);
    if (!res.ok) {
      const msg =
        formatErrorDetail(body.detail) ||
        body.message ||
        (body._raw ? `Unexpected response (${res.status}). Check API URL: ${getApiBase()}` : "") ||
        `Request failed (${res.status}). API: ${getApiBase()}`;
      err.textContent = msg;
      return;
    }
    const resultEl = document.getElementById("result");
    document.getElementById("result-empty").classList.add("hidden");
    resultEl.classList.remove("hidden", "result-positive", "result-negative", "result-popin");
    void resultEl.offsetWidth;
    resultEl.classList.add("result-popin", body.prediction === 1 ? "result-positive" : "result-negative");
    const predEl = document.getElementById("out-pred");
    const probEl = document.getElementById("out-prob");
    predEl.classList.remove("value--pop");
    probEl.classList.remove("value--pop");
    void predEl.offsetWidth;
    predEl.textContent = String(body.prediction);
    probEl.textContent = (body.probability_heart_disease * 100).toFixed(1) + "%";
    predEl.classList.add("value--pop");
    probEl.classList.add("value--pop");
    document.getElementById("out-msg").textContent = body.message;
    void pingApi();
  } catch (ex) {
    err.textContent = `Network error — open the app from the API (${getApiBase()}) or set <meta name="api-base" content="http://127.0.0.1:8000" />. (${ex?.message || ex})`;
  } finally {
    btn?.removeAttribute("disabled");
  }
});

loadMetrics();
loadFi();
pingApi();
