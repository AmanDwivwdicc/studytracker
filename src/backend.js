// Backend discovery for split deployments.
//
// Priority:
//   1. VITE_API_BASE_URL / VITE_WS_BASE_URL build-time variables
//   2. public/backend.json (runtime configuration, no rebuild of env vars)
//   3. same origin (frontend and backend served together)
const ENV_API = (import.meta.env.VITE_API_BASE_URL || "").trim().replace(/\/$/, "");
const ENV_WS = (import.meta.env.VITE_WS_BASE_URL || "").trim();

let apiBase = ENV_API;
let wsUrl = ENV_WS;
let initialized = false;
let pending = null;

async function loadRuntimeConfig() {
  try {
    const response = await fetch("/backend.json", { cache: "no-store" });
    if (!response.ok) return;
    const data = await response.json();
    if (!apiBase) apiBase = String(data.apiBaseUrl || "").trim().replace(/\/$/, "");
    if (!wsUrl) wsUrl = String(data.wsUrl || "").trim();
  } catch {
    /* no runtime configuration available - stay on the current origin */
  }
}

export function initBackend() {
  if (!pending) {
    pending = (async () => {
      initialized = true;
      if (!apiBase || !wsUrl) await loadRuntimeConfig();
      return apiBase;
    })();
  }
  return pending;
}

export function isInitialized() {
  return initialized;
}

export function backendBase() {
  return apiBase;
}

export function backendWsUrl() {
  if (wsUrl) return wsUrl;
  if (apiBase) return `${apiBase.replace(/^http/i, "ws").replace(/\/$/, "")}/ws`;
  if (typeof window !== "undefined") {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    return `${protocol}://${window.location.host}/ws`;
  }
  return "";
}

export function backendAssetBase() {
  return apiBase;
}
