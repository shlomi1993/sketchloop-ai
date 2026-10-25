# ADR 0015: Abstract base classes for replaceable components

Status: accepted

## Context

R02 and R08 require replaceable components behind documented interfaces. [ADR 0005](0005-domain-records-and-generator-contract.md), [ADR 0010](0010-sessions-and-evaluator.md), and [ADR 0014](0014-replaceable-components-and-settings.md) defined `Generator`, `Evaluator`, `CaptureSource`, `Preprocessor`, and `ExperimentStore` as `typing.Protocol` classes, so an implementation matched by having the right methods. The repository checks run Ruff and pytest but no static type checker, so nothing reported a mismatched implementation until a missing method was called during a round.

## Decision

The five component interfaces become abstract base classes (`abc.ABC` with `@abstractmethod`), and every implementation subclasses its interface: `FakeGenerator` and `DiffusersSketchGenerator` subclass `Generator`, `NoOpEvaluator` subclasses `Evaluator`, `FileCapture` and `CameraCapture` subclass `CaptureSource`, `StrokePreprocessor` subclasses `Preprocessor`, and `FolderStore` and `MemoryStore` subclass `ExperimentStore`. A researcher adding a component imports and subclasses the base class, which states the contract in one place and fails when the object is created if a method is missing.

Method names and signatures are unchanged. Amended: the two internal interfaces first kept as protocols are now abstract base classes too, so the project uses no `typing.Protocol`. `VideoSource` has `is_opened`, `read`, and `release`, and `OpenCVCamera` wraps `cv2.VideoCapture` to implement it. `PipelineRunner` is implemented by `DiffusersRunner`, which wraps the loaded pipeline. An object the project does not define is wrapped in a subclass rather than matched structurally.

Alternatives were keeping the protocols and adding a type checker such as Pyright or mypy to the checks, or doing both. The owner preferred an explicit base class that a new backend must import.

## Consequences

Every implementation, including test doubles, now imports its base class from the library, and a class with the right methods but no base class is no longer accepted by type checkers. Records, settings, and behavior are unchanged. Adding a static type checker remains a separate, optional decision.
