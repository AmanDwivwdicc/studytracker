// Small helper around the REST API of the Study Tracker server.
import { backendAssetBase, backendBase, backendWsUrl, initBackend } from "./backend.js";

async function request(path, options = {}) {
  await initBackend();
  const response = await fetch(`${backendBase()}/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const body = await response.json();
      message = body.error || message;
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }
  return response.json();
}

export const api = {
  state: () => request("/state"),
  settings: () => request("/settings"),
  saveSettings: (settings) =>
    request("/settings", { method: "PUT", body: JSON.stringify(settings) }),
  addCamera: (payload) => request("/cameras", { method: "POST", body: JSON.stringify(payload) }),
  removeCamera: (id) => request(`/cameras/${id}`, { method: "DELETE" }),
  pushFrame: async (id, blob) => {
    await initBackend();
    const response = await fetch(`${backendBase()}/api/cameras/${encodeURIComponent(id)}/frame`, {
      method: "POST",
      headers: { "Content-Type": blob.type || "image/jpeg" },
      body: blob,
    });
    if (!response.ok) throw new Error("Could not send the frame to the server");
    return response.json();
  },
  capabilities: () => request("/capabilities"),
  webcams: () => request("/webcams"),
  alerts: () => request("/alerts"),
  clearAlerts: () => request("/alerts", { method: "DELETE" }),
  audio: () => request("/audio"),
  uploadAudio: async (name, file) => {
    await initBackend();
    const response = await fetch(`${backendBase()}/api/audio/${encodeURIComponent(name)}`, {
      method: "POST",
      headers: { "Content-Type": file.type || "application/octet-stream" },
      body: await file.arrayBuffer(),
    });
    if (!response.ok) throw new Error("Could not upload the audio file");
    return response.json();
  },
};

export async function assetUrl(name) {
  await initBackend();
  return `${backendAssetBase()}/assets/${encodeURIComponent(name)}`;
}

export const apiBaseUrl = backendBase;
export const wsBaseUrl = backendWsUrl;
export { initBackend };
