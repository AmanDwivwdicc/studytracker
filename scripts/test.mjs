// End-to-end check: connect a webcam, receive frames over the WebSocket and
// verify that the phone detector runs on the real frames.
import WebSocket from "ws";

const BASE = "http://localhost:5174";

async function main() {
  console.log("1) state:", JSON.stringify((await (await fetch(`${BASE}/api/state`)).json()).cameras));

  const created = await (
    await fetch(`${BASE}/api/cameras`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "webcam", value: "0" }),
    })
  ).json();
  console.log("2) camera:", JSON.stringify(created.camera));
  const cameraId = created.camera.id;

  const ws = new WebSocket("ws://localhost:5174/ws");
  let frames = 0;
  let phoneMessages = 0;
  let lastBoxes = null;
  const statuses = [];

  await new Promise((resolve, reject) => {
    ws.on("open", resolve);
    ws.on("error", reject);
  });
  console.log("3) websocket connected");

  ws.on("message", (data, isBinary) => {
    if (isBinary) {
      frames += 1;
      return;
    }
    const msg = JSON.parse(data.toString());
    if (msg.type === "phone") {
      phoneMessages += 1;
      lastBoxes = msg.boxes;
    }
    if (msg.type === "camera-status") statuses.push(msg.camera.status);
  });

  await new Promise((r) => setTimeout(r, 15000));

  const state = await (await fetch(`${BASE}/api/state`)).json();
  console.log("4) frames received:", frames);
  console.log("5) phone detection messages:", phoneMessages, "boxes:", lastBoxes);
  console.log("6) camera statuses seen:", [...new Set(statuses)].join(","));
  console.log("7) camera list:", state.cameras.map((c) => `${c.name}=${c.status}@${c.fps}fps`).join(", "));
  console.log("8) detector:", JSON.stringify(state.detector));

  // Send an eye-state message to check the alert pipeline.
  const alertPromise = new Promise((resolve) => {
    const handler = (data, isBinary) => {
      if (isBinary) return;
      const msg = JSON.parse(data.toString());
      if (msg.type === "alert") {
        ws.off("message", handler);
        resolve(msg.event);
      }
    };
    ws.on("message", handler);
  });
  // Force a sleep alert by using a very short closure duration.
  await fetch(`${BASE}/api/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ min_eye_closure_duration: 0.2, sleep_alert_cooldown: 0 }),
  });
  // The browser reports the eye state continuously - do the same here.
  const sender = setInterval(() => {
    ws.send(JSON.stringify({ type: "eye-state", cameraId, cameraName: "test", closed: true }));
  }, 150);
  const event = await Promise.race([
    alertPromise,
    new Promise((r) => setTimeout(() => r(null), 6000)),
  ]);
  clearInterval(sender);
  console.log("9) alert event:", JSON.stringify(event));

  ws.close();
  await fetch(`${BASE}/api/cameras/${cameraId}`, { method: "DELETE" });
  console.log("DONE");
  process.exit(0);
}

main().catch((err) => {
  console.error("FAILED:", err);
  process.exit(1);
});
