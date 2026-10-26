// Start the webcam when the page loads, since Gradio cannot do it itself.
window.sketchloop = {
    // Press Gradio's webcam controls once they appear. Recheck the titles after Gradio upgrades.
    openWebcam(attemptsLeft = 80) {
        const grant = document.querySelector('[title="grant webcam access"] button');
        if (!grant) {
            if (attemptsLeft > 0) setTimeout(() => this.openWebcam(attemptsLeft - 1), 250);
            return;
        }

        grant.click();
        setTimeout(() => {
            const start = document.querySelector('[title="start recording"]');
            (start?.closest("button") ?? start)?.click();
        }, 1500);
    },
};

window.addEventListener("load", () => window.sketchloop.openWebcam());
