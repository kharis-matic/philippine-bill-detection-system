const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = document.querySelectorAll(".tab-panel");

tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
        tabButtons.forEach((item) => item.classList.remove("active"));
        tabPanels.forEach((panel) => panel.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.tab).classList.add("active");
    });
});

function setMessage(id, text, type = "") {
    const element = document.getElementById(id);
    element.textContent = text;
    element.className = `message ${type}`.trim();
}

function renderSummary(containerId, data) {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div><span>20-peso bills</span><strong>${data.count_20 ?? 0}</strong></div>
        <div><span>50-peso bills</span><strong>${data.count_50 ?? 0}</strong></div>
        <div><span>Total value</span><strong>PHP ${data.total_value ?? 0}</strong></div>
    `;
}

function renderDetections(containerId, data, limit = 20) {
    const container = document.getElementById(containerId);
    const detections = data.detections || [];

    if (!detections.length) {
        container.innerHTML = `<div class="detection-item"><strong>No bill detected</strong><span>${data.message || ""}</span></div>`;
        return;
    }

    const visibleDetections = detections.slice(0, limit);
    container.innerHTML = visibleDetections.map((item) => `
        <div class="detection-item">
            <strong>${item.class_name}</strong>
            <span>PHP ${item.denomination} | ${(item.confidence * 100).toFixed(1)}%</span>
        </div>
    `).join("");

    if (detections.length > limit) {
        container.innerHTML += `
            <div class="detection-item">
                <strong>${detections.length - limit} more detections</strong>
                <span>Shown in processed output</span>
            </div>
        `;
    }
}

async function submitForm(form, endpoint, messageId) {
    const formData = new FormData(form);
    setMessage(messageId, "Processing...", "");

    const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || "Detection failed.");
    }

    return data;
}

document.getElementById("image-form").addEventListener("submit", async (event) => {
    event.preventDefault();

    try {
        const data = await submitForm(event.currentTarget, "/detect_image", "image-message");
        const resultImage = document.getElementById("image-result");
        resultImage.src = `${data.image_url}?t=${Date.now()}`;
        resultImage.classList.remove("hidden");
        renderSummary("image-summary", data);
        renderDetections("image-detections", data);
        setMessage("image-message", data.message || "Image detection complete.", "success");
    } catch (error) {
        setMessage("image-message", error.message, "error");
    }
});

document.getElementById("video-form").addEventListener("submit", async (event) => {
    event.preventDefault();

    try {
        const data = await submitForm(event.currentTarget, "/detect_video", "video-message");
        const resultVideo = document.getElementById("video-result");
        resultVideo.src = `${data.video_url}?t=${Date.now()}`;
        resultVideo.classList.remove("hidden");
        renderSummary("video-summary", data);
        renderDetections("video-detections", data);
        const frameText = data.frame_count ? ` Processed ${data.frame_count} frames.` : "";
        setMessage("video-message", `${data.message || "Video processing complete."}${frameText}`, "success");
    } catch (error) {
        setMessage("video-message", error.message, "error");
    }
});

document.getElementById("start-webcam").addEventListener("click", () => {
    const stream = document.getElementById("webcam-stream");
    stream.src = `/video_feed?t=${Date.now()}`;
    stream.classList.remove("hidden");
    setMessage("webcam-message", "Webcam detection started.", "success");
});

document.getElementById("stop-webcam").addEventListener("click", async () => {
    try {
        await fetch("/stop_webcam", { method: "POST" });
        const stream = document.getElementById("webcam-stream");
        stream.removeAttribute("src");
        stream.classList.add("hidden");
        setMessage("webcam-message", "Webcam stopped.", "success");
    } catch (error) {
        setMessage("webcam-message", "Could not stop webcam cleanly.", "error");
    }
});
