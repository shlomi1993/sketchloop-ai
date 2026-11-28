# ADR 0029: Sketch orientation controls

Status: accepted

## Context

The owner requested a centered page capped at 1440 px and Landscape/Portrait controls beside the camera settings. Capture and presentation serve R01 and R07, while S04 requires preserved drawing data. Camera rotation is independent of paper canvas dimensions. The controller already keeps a configured canvas size, and each drawing records its width and height.

## Decision

Cap the page width with CSS. Add a Landscape/Portrait radio selector styled consistently with the existing selection controls. Reorder the configured side lengths rather than introducing a new resolution or rotating captured images. Keep square canvases square.

Expose a controller operation that validates the orientation and refuses a change while a drawing is open. Apply UI changes through the same queue as Start and Finish. Lock the selector during drawings, restore it on Finish and reload, and refresh it across tabs. Preview and the next drawing use the selected dimensions, while completed records retain theirs.

## Consequences

No settings key, stored field, schema migration, dependency, or page is added. Camera rotation continues working independently. The selected orientation is held in the running controller and returns to configured dimensions after a server restart. The stroke model warms up only at the configured startup size, so the first request at a swapped size may take longer. Existing tests are extended without adding test functions. Real camera and model validation in both orientations remains part of the pilot.
