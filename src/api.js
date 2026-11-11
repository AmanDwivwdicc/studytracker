// Small helper around the REST API of the Study Tracker server.
const API = "/api";

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
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
  webcams: () => request("/webcams"),
  alerts: () => request("/alerts"),
  clearAlerts: () => request("/alerts", { method: "DELETE" }),
  audio: () => request("/audio"),
  uploadAudio: async (name, file) => {
    const response = await fetch(`${API}/audio/${encodeURIComponent(name)}`, {
      method: "POST",
      headers: { "Content-Type": file.type || "application/octet-stream" },
      body: await file.arrayBuffer(),
    });
    if (!response.ok) throw new Error("Could not upload the audio file");
    return response.json();
  },
};
