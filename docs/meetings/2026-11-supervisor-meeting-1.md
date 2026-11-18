# Supervisor meeting 1

Date: November 2026. Attendees: the owner and the project supervisor. The meeting followed the progress page: literature, plan against execution, project structure, models, a live demo, the frontend and backend split, and next steps.

This note records what was said and decided. [ADR 0021](../decisions/0021-guided-sketching-and-user-study.md) turns it into a scope decision, [PROJECT.md](../PROJECT.md) lists the resulting requirements S01 to S05, and [ROADMAP.md](../ROADMAP.md) holds the tasks.

## Key insights

- **Interaction over generation:** the project should focus on human-AI interaction, meaning how the system guides the creative process, rather than on what the AI can create by itself.
- **Suggest the next strokes:** instead of generating a complete image from the sketch, the system should suggest how to continue the sketch.
- **Implicit acceptance:** whether the person accepts a suggestion is inferred from whether they follow it, with no explicit accept or reject action.
- **Example:** the person states in the prompt that they are designing a car and starts with a rectangle. The system shows visual suggestions for adding wheels or a windshield. If the person then draws the wheels, that suggestion counts as accepted, and otherwise as rejected.
- **Guide, do not design:** the system guides the person throughout the design process rather than producing the design for them.

## Decisions

1. The primary objective of the implementation is a working system that supports the user study below, including its data collection.
2. Guided sketching, with suggestions inferred as accepted or rejected, becomes the main interaction. Complete-image generation stays in the system but is no longer the focus.
3. T07 and T08 continue to be refined, and the evaluation and documentation planned as T09 is still to be done. The roadmap now splits it between T08 and the documentation task T12.
4. Still required: test the system's modularity by choosing a new model and integrating it, continue the next round from a selected candidate (U2), and store prompt suggestions in the experiment records.

## User study

| Item | Agreed |
| --- | --- |
| Tasks | Four simple design tasks: a car, a motorcycle, a ship, and an airplane |
| Participants | Five |
| Conditions | Every participant completes all four tasks both with the system and without it, meaning with and without AI |
| Output | 5 participants × 4 tasks × 2 conditions = 40 drawings |
| After collection | Schedule a meeting with the supervisor to discuss the next steps |
| Evaluation | The supervisor submits the drawings for expert evaluation |
| Target | A conference submission in early 2027 |

## Later steps

- **Final report:** write a final report. The supervisor will provide the guidelines later.
- **Thesis extension:** include professional artists in the study and compare directly. Collect 40 more drawings from experts and compare them with the drawings of inexperienced participants.

## Action items

| # | Who | Action | Tracked in |
| --- | --- | --- | --- |
| 1 | Owner | Update the implementation plan for the new direction | [ROADMAP.md](../ROADMAP.md) |
| 2 | Owner | Choose and build next-stroke suggestions with implicit acceptance | T09 |
| 3 | Owner | Build the study mode and the data export | T10 |
| 4 | Owner | Pilot the full protocol, then recruit five participants and collect the 40 drawings | T10, T11 |
| 5 | Owner | Schedule the follow-up meeting once the 40 drawings are collected | T11 |
| 6 | Owner | Refine T07 and T08, and write the project documentation | T07, T08, T12 |
| 7 | Owner | Integrate a new model, build U2, and store prompt suggestions in the records | T07, U2, U1 |
| 8 | Supervisor | Send the drawings for expert evaluation and lead the submission | Outside the repository |
| 9 | Supervisor | Provide the final report guidelines | T12 |

