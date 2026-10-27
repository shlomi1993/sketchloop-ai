// Start the webcam stream on load and whenever it is shown again, since Gradio cannot do it itself.
window.sketchloop = {
    // Press Gradio's grant and record controls until the stream runs. Recheck the selectors after Gradio upgrades.
    startWebcam(attemptsLeft = 80) {
        const webcam = document.getElementById("webcam");
        if (!webcam || webcam.querySelector('[title="stop recording"]')) return;

        // Wait longer after a grant, so the permission prompt is not asked for twice.
        const grant = webcam.querySelector('[title="grant webcam access"] button');
        const record = webcam.querySelector('[title="start recording"]')?.closest("button");
        (grant ?? record)?.click();
        if (attemptsLeft > 0) setTimeout(() => this.startWebcam(attemptsLeft - 1), grant ? 1000 : 250);
    },

    // Start the stream each time the webcam input becomes visible, including the first time.
    watchWebcam() {
        let wasVisible = false;
        setInterval(() => {
            const isVisible = Boolean(document.getElementById("webcam")?.offsetParent);
            if (isVisible && !wasVisible) this.startWebcam();
            wasVisible = isVisible;
        }, 500);
    },
};

// Gradio adds this script after the page's load event, so start watching now.
window.sketchloop.watchWebcam();
