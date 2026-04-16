console.log("JS LOADED");

document.addEventListener("DOMContentLoaded", () => {
  "use strict";

  const API_BASE = "http://127.0.0.1:5000";

  /** @type {HTMLDivElement} */
  const dropzone = document.getElementById("dropzone");
  /** @type {HTMLInputElement} */
  const fileInput = document.getElementById("fileInput");
  /** @type {HTMLDivElement} */
  const fileMeta = document.getElementById("fileMeta");

  /** @type {HTMLButtonElement} */
  const analyzeBtn = document.getElementById("analyzeBtn");
  /** @type {HTMLButtonElement} */
  const abortBtn = document.getElementById("abortBtn");
  /** @type {HTMLButtonElement} */
  const clearBtn = document.getElementById("clearBtn");
  /** @type {HTMLButtonElement} */
  const clearLogBtn = document.getElementById("clearLogBtn");
  /** @type {HTMLButtonElement} */
  const copyLogBtn = document.getElementById("copyLogBtn");

  /** @type {HTMLImageElement} */
  const previewImg = document.getElementById("previewImg");
  /** @type {HTMLDivElement} */
  const previewFrame = document.getElementById("previewFrame");

  /** @type {HTMLDivElement} */
  const terminalBody = document.getElementById("terminalBody");

  /** @type {HTMLDivElement} */
  const backendPill = document.getElementById("backendPill");
  /** @type {HTMLSpanElement} */
  const backendText = document.getElementById("backendText");

  /** @type {HTMLDivElement} */
  const terminalView = document.getElementById("terminalView");
  /** @type {HTMLDivElement} */
  const resultView = document.getElementById("resultView");

  /** @type {HTMLDivElement} */
  const outputFrame = document.getElementById("outputFrame");
  /** @type {HTMLImageElement} */
  const outputImg = document.getElementById("outputImg");
  /** @type {HTMLDivElement} */
  const outputMeta = document.getElementById("outputMeta");
  /** @type {HTMLDivElement} */
  const scanbar = document.getElementById("scanbar");

  /** @type {HTMLDivElement} */
  const angleValue = document.getElementById("angleValue");
  /** @type {HTMLDivElement} */
  const confidenceValue = document.getElementById("confidenceValue");
  /** @type {HTMLDivElement} */
  const conclusionValue = document.getElementById("conclusionValue");
  /** @type {HTMLDivElement} */
  const plainConclusionValue = document.getElementById("plainConclusionValue");

  /** @type {HTMLDivElement} */
  const alertBox = document.getElementById("alertBox");
  /** @type {HTMLDivElement} */
  const alertTitle = document.getElementById("alertTitle");
  /** @type {HTMLDivElement} */
  const alertBody = document.getElementById("alertBody");

  /** @type {File | null} */
  let selectedFile = null;
  /** @type {string | null} */
  let previewObjectUrl = null;

  /** @type {AbortController | null} */
  let activeController = null;
  /** @type {boolean} */
  let isBusy = false;

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function setBackendState(state, detailText) {
    backendPill.dataset.state = state; 
    const label = detailText ?? (state === "ok" ? "BACKEND: ONLINE" : state === "down" ? "BACKEND: OFFLINE" : "BACKEND: UNKNOWN");
    backendText.textContent = label;
  }

  function showAlert(title, body) {
    alertTitle.textContent = title;
    alertBody.textContent = body;
    alertBox.hidden = false;
  }

  function clearAlert() {
    alertBox.hidden = true;
    alertTitle.textContent = "Error";
    alertBody.textContent = "—";
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return "—";
    const units = ["B", "KB", "MB", "GB"];
    let n = bytes;
    let i = 0;
    while (n >= 1024 && i < units.length - 1) {
      n /= 1024;
      i += 1;
    }
    return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
  }

  function sanitizeFilename(name) {
    return String(name || "unknown").replace(/[^\w.\- ()]/g, "_");
  }

  function terminalClear() {
    terminalBody.innerHTML = "";
  }

  function terminalAppendLine(text, tone = "dim") {
    const div = document.createElement("div");
    div.className = `term-line ${tone === "ok" ? "term-line--ok" : tone === "danger" ? "term-line--danger" : "term-line--dim"}`;
    div.textContent = text;
    terminalBody.appendChild(div);
    terminalBody.scrollTop = terminalBody.scrollHeight;
    return div;
  }

  async function terminalTypeLine(prefix, message, tone = "dim", signal) {
    const line = terminalAppendLine(`${prefix}`, tone);
    const full = `${prefix}${message}`;
    line.textContent = prefix;

    for (let i = prefix.length; i <= full.length; i += 1) {
      if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
      line.textContent = full.slice(0, i);
      await sleep(8 + Math.random() * 14);
    }
  }

  function setBusy(nextBusy) {
    isBusy = nextBusy;
    analyzeBtn.disabled = nextBusy || !selectedFile;
    abortBtn.disabled = !nextBusy;
    clearBtn.disabled = nextBusy || !selectedFile;
    fileInput.disabled = nextBusy;
    dropzone.classList.toggle("is-disabled", nextBusy);
  }

  function setPreview(file) {
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
      previewObjectUrl = null;
    }

    selectedFile = file;
    console.log("FILE SELECTED:", selectedFile);
    previewObjectUrl = URL.createObjectURL(file);
    previewImg.src = previewObjectUrl;
    previewFrame.classList.add("has-image");

    fileMeta.textContent = `${sanitizeFilename(file.name)} • ${formatBytes(file.size)} • ${file.type || "unknown/unknown"}`;
    terminalAppendLine(`[intake] loaded: ${sanitizeFilename(file.name)} (${formatBytes(file.size)})`, "ok");

    analyzeBtn.disabled = false;
    clearBtn.disabled = false;
    clearAlert();
  }

  function clearSelection() {
    selectedFile = null;
    fileInput.value = "";
    previewImg.removeAttribute("src");
    previewFrame.classList.remove("has-image");
    fileMeta.textContent = "No file loaded.";
    analyzeBtn.disabled = true;
    clearBtn.disabled = true;
  }

  function resetOutput() {
    outputImg.removeAttribute("src");
    outputFrame.classList.remove("has-image");
    outputFrame.classList.remove("is-scanning");
    outputMeta.textContent = "No analysis yet.";
    angleValue.textContent = "—";
    confidenceValue.textContent = "—";
    conclusionValue.textContent = "Awaiting evidence.";
    plainConclusionValue.textContent = "Upload and analyze an image to see a simple explanation.";
  }

  function showTerminalView() {
    terminalView.classList.add("is-active");
    resultView.classList.remove("is-active");
  }

  function showResultView() {
    resultView.classList.add("is-active");
    terminalView.classList.remove("is-active");
  }

  function isAllowedImage(file) {
    const type = String(file.type || "").toLowerCase();
    if (type.startsWith("image/")) return true;
    const name = String(file.name || "").toLowerCase();
    return /\.(png|jpe?g|webp|bmp|gif|tiff?)$/.test(name);
  }

  function normalizeConfidence(value) {
    if (!Number.isFinite(value)) return null;
    if (value <= 1) return Math.max(0, Math.min(100, value * 100));
    return Math.max(0, Math.min(100, value));
  }

  function conclusionFromAngle(angleDeg) {
    if (!Number.isFinite(angleDeg)) return "Conclusion unavailable (invalid angle).";
    const a = Math.max(0, Math.min(90, angleDeg));

    if (a < 15) {
      return "Very shallow (grazing) impact: low-angle trajectory; likely lateral source or low-height cast-off.";
    }
    if (a < 35) {
      return "Low to medium height trajectory: consistent with oblique impact and moderate forward motion.";
    }
    if (a < 55) {
      return "Medium trajectory: balanced vertical and horizontal components; consistent with mid-height source region.";
    }
    if (a < 75) {
      return "High-angle trajectory: closer to perpendicular impact; indicates higher source height or steeper descent.";
    }
    return "Near-perpendicular impact: circular tendency; likely overhead origin with minimal lateral component.";
  }

  function plainConclusionFromAngle(angleDeg) {
    if (!Number.isFinite(angleDeg)) return "Could not determine the blood drop direction from this image.";
    const a = Math.max(0, Math.min(90, angleDeg));
    if (a < 15) return "The blood likely hit the surface at a very shallow side angle.";
    if (a < 35) return "The blood likely came in from the side at a low-to-medium height.";
    if (a < 55) return "The blood likely hit from a medium angle and medium height.";
    if (a < 75) return "The blood likely came from higher up and hit more directly.";
    return "The blood likely fell almost straight down onto the surface.";
  }

  async function fetchWithTimeout(url, options = {}, timeoutMs = 12000) {
    const controller = options.signal ? null : new AbortController();
    const signal = options.signal ?? controller.signal;
    const timer = setTimeout(() => controller?.abort(), timeoutMs);

    try {
      return await fetch(url, { ...options, signal });
    } finally {
      clearTimeout(timer);
    }
  }

  async function checkBackendReachable() {
    try {
      const res = await fetchWithTimeout(`${API_BASE}/image/__healthcheck__`, { method: "GET" }, 2500);
      void res;
      setBackendState("ok", "BACKEND: ONLINE");
      return true;
    } catch {
      setBackendState("down", "BACKEND: OFFLINE");
      return false;
    }
  }

  async function runTerminalSimulation(signal) {
    const steps = [
      ["[scan] ", "initializing photometric normalization…"],
      ["[scan] ", "detecting stain boundaries (contours)…"],
      ["[scan] ", "fitting ellipse parameters (major/minor axes)…"],
      ["[scan] ", "estimating angle via ratio model…"],
      ["[scan] ", "validating measurement stability…"],
      ["[scan] ", "rendering overlay + packaging output…"],
    ];

    const baseDelay = 240;
    for (let i = 0; i < steps.length; i += 1) {
      const [p, msg] = steps[i];
      await terminalTypeLine(p, msg, "dim", signal);
      await sleep(baseDelay + Math.random() * 170);
    }
  }

  async function analyzeImage(file) {
    if (!file) {
      showAlert("Upload image first", "Please select or drop an image before running analysis.");
      terminalAppendLine("[error] upload image first.", "danger");
      return;
    }
    if (!isAllowedImage(file)) {
      terminalAppendLine(`[error] invalid file type: ${file.type || "unknown"}`, "danger");
      showAlert("Invalid file", "Please upload a valid image file (JPG, PNG, WEBP, etc.).");
      return;
    }

    clearAlert();
    resetOutput();
    showTerminalView();

    const reachable = await checkBackendReachable();
    if (!reachable) {
      terminalAppendLine("[net] backend unreachable at 127.0.0.1:5000", "danger");
      showAlert("Backend unreachable", "Start your Flask server on http://127.0.0.1:5000, then retry.");
      return;
    }

    setBusy(true);
    terminalAppendLine(`[run] submitting evidence: ${sanitizeFilename(file.name)}`, "ok");

    activeController = new AbortController();
    const { signal } = activeController;

    outputFrame.classList.add("is-scanning");

    try {
      await runTerminalSimulation(signal);

      const form = new FormData();
      form.append("image", file);
      console.log("SENDING REQUEST");

      terminalAppendLine("[api] POST /analyze", "dim");

      const res = await fetchWithTimeout(
        `${API_BASE}/analyze`,
        {
          method: "POST",
          body: form,
          signal,
        },
        15000
      );

      if (!res.ok) {
        let message = `Backend responded ${res.status} ${res.statusText}`;
        try {
          const payload = await res.json();
          if (payload?.error) {
            message = String(payload.error);
          }
        } catch {
          const text = await res.text().catch(() => "");
          if (text) {
            message = `${message} — ${text.slice(0, 160)}`;
          }
        }
        throw new Error(message);
      }

      /** @type {{angle:number, confidence:number, output_image:string}} */
      const data = await res.json();

      const angle = Number(data?.angle);
      const conf = normalizeConfidence(Number(data?.confidence));
      const outName = String(data?.output_image || "");

      if (!outName) {
        throw new Error("Backend did not return output_image filename.");
      }

      const angleText = Number.isFinite(angle) ? `${angle.toFixed(1)}` : "—";
      const confText = conf == null ? "—" : `${conf.toFixed(0)}`;

      angleValue.textContent = angleText;
      confidenceValue.textContent = confText === "—" ? "—" : `${confText}%`;
      conclusionValue.textContent = conclusionFromAngle(angle);
      plainConclusionValue.textContent = plainConclusionFromAngle(angle);

      const outputUrl = `${API_BASE}/image/${encodeURIComponent(data.output_image)}?t=${Date.now()}`;


outputImg.onload = null;
outputImg.onerror = null;


outputImg.src = outputUrl;

outputFrame.classList.add("has-image");
outputMeta.textContent = `output: ${sanitizeFilename(outName)}`;


outputImg.onload = () => {
  terminalAppendLine(`[ok] output received: ${sanitizeFilename(outName)}`, "ok");
  showResultView();
};

outputImg.onerror = () => {
  terminalAppendLine("[warn] output image failed to load", "danger");
  showAlert("Output fetch failed", "Image could not be loaded.");
};

      
      outputImg.onload = () => {
        terminalAppendLine(`[ok] output received: ${sanitizeFilename(outName)}`, "ok");
        showResultView();
      };
      outputImg.onerror = () => {
        terminalAppendLine("[warn] output image failed to load from /image/<filename>", "danger");
        showAlert("Output fetch failed", "Backend returned a filename, but the processed image could not be loaded.");
      };
      
      
    } catch (err) {
      if (err?.name === "AbortError") {
        terminalAppendLine("[abort] analysis cancelled by operator.", "danger");
        showAlert("Aborted", "Analysis aborted.");
      } else {
        terminalAppendLine(`[error] ${String(err?.message || err)}`, "danger");
        showAlert("Analysis failed", String(err?.message || err));
      }
      
      await checkBackendReachable();
    } finally {
      outputFrame.classList.remove("is-scanning");
  activeController = null;

  
  setTimeout(() => setBusy(false), 200);
    }
  }

  function abortActive() {
    if (!activeController) return;
    activeController.abort();
  }

  function bindIntakeEvents() {
    window.addEventListener("dragover", (e) => {
      e.preventDefault();
    });
    window.addEventListener("drop", (e) => {
      e.preventDefault();
    });

    dropzone.addEventListener("click", () => {
      if (isBusy) return;
      fileInput.click();
    });

    dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });

    const mark = (on) => dropzone.classList.toggle("is-dragover", on);

    dropzone.addEventListener("dragenter", (e) => {
      e.preventDefault();
      mark(true);
    });
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      mark(true);
    });
    dropzone.addEventListener("dragleave", (e) => {
      e.preventDefault();
      mark(false);
    });
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      mark(false);
      if (isBusy) return;

      const file = e.dataTransfer?.files?.[0];
      if (!file) return;
      resetOutput();
      setPreview(file);
      analyzeBtn.disabled = false;
    });

    fileInput.addEventListener("change", () => {
      if (isBusy) return;
      const file = fileInput.files?.[0] ?? null;
      if (!file) return;
      resetOutput();
      setPreview(file);
      analyzeBtn.disabled = false;
    });
  }

  function bindControls() {
    analyzeBtn.addEventListener("click", (e) => {
      e.preventDefault();
      console.log("CLICK WORKING");
      console.log("Analyze clicked");
      analyzeImage(selectedFile);
    });
    abortBtn.addEventListener("click", abortActive);

    clearBtn.addEventListener("click", () => {
      if (isBusy) return;
      clearSelection();
      resetOutput();
      terminalAppendLine("[intake] cleared.", "dim");
      clearAlert();
    });

    clearLogBtn.addEventListener("click", () => {
      terminalClear();
      terminalAppendLine("[boot] console ready. load evidence to begin.", "dim");
    });

    copyLogBtn.addEventListener("click", async () => {
      const text = Array.from(terminalBody.querySelectorAll(".term-line"))
        .map((el) => el.textContent ?? "")
        .join("\n");
      try {
        await navigator.clipboard.writeText(text);
        terminalAppendLine("[ui] terminal copied to clipboard.", "ok");
      } catch {
        terminalAppendLine("[ui] clipboard unavailable in this context.", "danger");
        showAlert("Clipboard blocked", "Your browser prevented clipboard access. Copy manually from the terminal window.");
      }
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (isBusy) abortActive();
      }
      if (e.key === "Enter") {
        e.preventDefault();
        if (document.activeElement && ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
        console.log("CLICK WORKING");
        analyzeImage(selectedFile);
      }
    });
  }

  function boot() {
    resetOutput();
    setBackendState("unknown", "BACKEND: UNKNOWN");
    bindIntakeEvents();
    bindControls();
    showTerminalView();
    checkBackendReachable().then((ok) => {
      terminalAppendLine(ok ? "[net] backend online." : "[net] backend offline (start Flask).", ok ? "ok" : "danger");
    });
  }

  boot();
});
