# Study guide

This guide explains how to run the user study. Five participants each draw four tasks, once with AI and once without, for 40 drawings in total.

## Before the study

1. Download the models once with `python scripts/download_models.py`.
2. Mount the webcam above the desk, facing down at an A4 sheet in landscape.
3. Give each participant a code such as `p1`. Never type real names into the system.
4. Alternate between participants which condition comes first.
5. Run a pilot with one AI drawing and one no-AI drawing.

## Start the page

```sh
sketchloop start --config study.toml
```

Wait until the terminal says next-stroke suggestions are ready, about 30 s. Fix any yellow warning before you start.

## Each drawing

1. Give the participant a fresh sheet and tell them the task.
2. Open the Experimenter panel and check that the camera shows the whole sheet.
3. Enter a name such as `p1_car_ai`, choose With AI or Without AI, and press Start.
4. The participant draws. In the AI condition they can ask for suggestions.
5. When they are done, press Finish twice.

Use one browser tab only. Reloading the page is safe. If an amber "check needed" alert appears, fix the camera before continuing.

## After the study

1. Run `sketchloop study summary` and check that all drawings are finished.
2. Run `sketchloop study export`. It writes the final drawings under random codes to `data/study-exports/`.
3. Send the experts only the `drawings/` folder. Keep `key.csv` private, because it maps the codes back to participants and conditions.
