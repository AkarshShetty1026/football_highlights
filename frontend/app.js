const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzone-text");
const fileInput = document.getElementById("file-input");
const fileNameEl = document.getElementById("file-name");
const submitBtn = document.getElementById("submit-btn");
const form = document.getElementById("upload-form");
const sensitivity = document.getElementById("sensitivity");
const sensitivityOut = document.getElementById("sensitivity-out");

const uploadCard = document.getElementById("upload-card");
const progressCard = document.getElementById("progress-card");
const progressTitle = document.getElementById("progress-title");
const progressFill = document.getElementById("progress-fill");
const progressDetail = document.getElementById("progress-detail");
const errorCard = document.getElementById("error-card");
const errorDetail = document.getElementById("error-detail");
const resultCard = document.getElementById("result-card");
const resultVideo = document.getElementById("result-video");
const resultStats = document.getElementById("result-stats");
const downloadVideo = document.getElementById("download-video");
const downloadTranscript = document.getElementById("download-transcript");
const downloadChapters = document.getElementById("download-chapters");
const indexSection = document.getElementById("index-section");
const indexSummary = document.getElementById("index-summary");
const chapterList = document.getElementById("chapter-list");
const resetBtn = document.getElementById("reset-btn");

const CATEGORY_ORDER = ["goal", "foul", "half_time", "full_time", "other"];

let selectedFile = null;
let pollTimer = null;

sensitivity.addEventListener("input", () => {
  sensitivityOut.textContent = Number(sensitivity.value).toFixed(2);
});

dropzone.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files?.[0];
  if (file) setSelectedFile(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) setSelectedFile(file);
});

function setSelectedFile(file) {
  selectedFile = file;
  fileNameEl.textContent = `${file.name} (${(file.size / (1024 ** 3)).toFixed(2)} GB)`;
  dropzoneText.textContent = "Looks good. Drop a different file to replace it.";
  submitBtn.disabled = false;
}

function showOnly(section) {
  [progressCard, errorCard, resultCard].forEach((el) => el.classList.add("hidden"));
  if (section) section.classList.remove("hidden");
}

function setProgress(fraction, title, detail) {
  // fraction is omitted when only the message changes (e.g. while reconnecting)
  if (Number.isFinite(fraction)) {
    progressFill.style.width = `${Math.round(fraction * 100)}%`;
  }
  if (title) progressTitle.textContent = title;
  if (detail !== undefined) progressDetail.textContent = detail;
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (!selectedFile) return;
  startUpload();
});

function startUpload() {
  uploadCard.classList.add("hidden");
  showOnly(progressCard);
  setProgress(0, "Uploading…", "0%");

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("target_minutes", document.getElementById("target-minutes").value);
  formData.append("sensitivity", sensitivity.value);
  formData.append("model_size", document.getElementById("model-size").value);
  formData.append("hotwords", document.getElementById("hotwords").value);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");

  xhr.upload.addEventListener("progress", (e) => {
    if (!e.lengthComputable) return;
    const frac = e.loaded / e.total;
    setProgress(frac, "Uploading…", `${Math.round(frac * 100)}%`);
  });

  xhr.addEventListener("load", () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      const data = JSON.parse(xhr.responseText);
      pollStatus(data.job_id);
    } else {
      showError(extractError(xhr));
    }
  });

  xhr.addEventListener("error", () => showError("Upload failed (network error)."));
  xhr.send(formData);
}

function extractError(xhr) {
  try {
    return JSON.parse(xhr.responseText).detail || xhr.statusText;
  } catch {
    return xhr.statusText || "Unknown error";
  }
}

// A job runs for half an hour. Over that many requests a transient "Failed to
// fetch" is close to inevitable, and it says nothing about the job -- which
// keeps running on the server regardless. Only give up after several
// consecutive failures, and show the job id so a lost session is recoverable.
const MAX_POLL_FAILURES = 8;

function pollStatus(jobId) {
  setProgress(0, "Processing…", "Starting…");
  let failures = 0;

  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/status/${jobId}`);
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data = await res.json();
      failures = 0;

      if (data.status === "error") {
        clearInterval(pollTimer);
        showError(data.error || "Processing failed.");
        return;
      }

      if (data.status === "done") {
        clearInterval(pollTimer);
        showResult(jobId, data.result);
        return;
      }

      setProgress(data.progress, "Processing…", data.stage_label);
    } catch (err) {
      failures += 1;
      if (failures >= MAX_POLL_FAILURES) {
        clearInterval(pollTimer);
        showError(
          `Lost contact with the server after ${MAX_POLL_FAILURES} tries (${err}). ` +
          `The job may still be running — reopen it at ${window.location.origin}/?job=${jobId}`
        );
        return;
      }
      setProgress(
        undefined,
        "Processing…",
        `Reconnecting… (${failures}/${MAX_POLL_FAILURES})`
      );
    }
  }, 1500);
}

function showError(message) {
  errorDetail.textContent = message;
  showOnly(errorCard);
}

function showResult(jobId, result) {
  resultVideo.src = `/api/download/${jobId}/video`;
  downloadVideo.href = `/api/download/${jobId}/video`;
  downloadTranscript.href = `/api/download/${jobId}/transcript`;
  downloadChapters.href = `/api/download/${jobId}/chapters`;

  if (result) {
    const src = formatDuration(result.source_duration_sec);
    const hl = formatDuration(result.highlight_duration_sec);
    resultStats.textContent = `${result.clip_count} clips · ${hl} highlight reel from a ${src} recording`;
    renderIndex(result.chapters || []);
  }

  showOnly(resultCard);
}

function renderIndex(chapters) {
  chapterList.innerHTML = "";
  if (!chapters.length) {
    indexSection.classList.add("hidden");
    return;
  }

  const counts = new Map();
  chapters.forEach((c) => counts.set(c.category, (counts.get(c.category) || 0) + 1));
  indexSummary.textContent = CATEGORY_ORDER.filter((cat) => counts.has(cat))
    .map((cat) => `${counts.get(cat)} ${labelFor(cat, counts.get(cat))}`)
    .join(" · ");

  chapters.forEach((chapter) => {
    const li = document.createElement("li");
    li.className = `chapter cat-${chapter.category}`;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "chapter-btn";
    // Seeking the <video> is why the reel timestamp is indexed alongside the
    // match clock -- the two are different timelines.
    button.addEventListener("click", () => {
      resultVideo.currentTime = chapter.reel_start_sec;
      resultVideo.play().catch(() => {});
      resultVideo.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });

    const time = document.createElement("span");
    time.className = "chapter-time";
    time.textContent = formatClock(chapter.reel_start_sec);

    const body = document.createElement("span");
    body.className = "chapter-body";

    const heading = document.createElement("span");
    heading.className = "chapter-label";
    heading.textContent = chapter.label;

    const clock = document.createElement("span");
    clock.className = "chapter-match";
    clock.textContent = `match ${chapter.match_clock}`;

    const reason = document.createElement("span");
    reason.className = "chapter-reason";
    reason.textContent = chapter.reason || "";

    body.append(heading, clock, reason);
    button.append(time, body);
    li.append(button);
    chapterList.append(li);
  });

  indexSection.classList.remove("hidden");
}

function labelFor(category, count) {
  const names = {
    goal: ["goal", "goals"],
    foul: ["foul", "fouls"],
    half_time: ["half time", "half time"],
    full_time: ["full time", "full time"],
    other: ["other highlight", "other highlights"],
  }[category] || ["moment", "moments"];
  return count === 1 ? names[0] : names[1];
}

function formatClock(sec) {
  sec = Math.max(Math.round(sec || 0), 0);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  const mm = h > 0 ? String(m).padStart(2, "0") : String(m);
  return h > 0 ? `${h}:${mm}:${String(s).padStart(2, "0")}` : `${mm}:${String(s).padStart(2, "0")}`;
}

function formatDuration(sec) {
  sec = Math.round(sec || 0);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  return h > 0 ? `${h}h ${m}m` : `${m}m ${s}s`;
}

// Jobs outlive the page: a half-hour run can lose its tab to a reload, a
// closed laptop, or a dropped connection. ?job=<id> reattaches to one that is
// still running, or opens one that already finished.
const requestedJob = new URLSearchParams(window.location.search).get("job");
if (requestedJob) {
  uploadCard.classList.add("hidden");
  showOnly(progressCard);
  pollStatus(requestedJob);
}

resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  chapterList.innerHTML = "";
  indexSection.classList.add("hidden");
  fileNameEl.textContent = "";
  dropzoneText.textContent = "Drag & drop a match video here, or click to choose a file";
  submitBtn.disabled = true;
  showOnly(null);
  uploadCard.classList.remove("hidden");
});
