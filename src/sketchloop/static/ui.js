// Helpers that the Gradio events in ui.py call.
window.sketchloop = {
    // Press Gradio's webcam controls, since Gradio cannot start the webcam itself. Recheck after Gradio upgrades.
    openWebcam() {
        const press = (selector) => {
            const element = document.querySelector(selector);
            (element?.closest("button") ?? element)?.click();
        };
        setTimeout(() => press('[title="grant webcam access"] button'), 500);
        setTimeout(() => press('[title="start recording"]'), 2000);
    },

    // Turn only the live preview, not the saved frame.
    rotatePreview(degrees) {
        document.querySelector("#webcam")?.style.setProperty("--sketch-rotation", `${degrees || 0}deg`);
    },
};
