# Dataset Card

## Source

Primary intended public source: Schema-Guided Dialogue, converted into agentic tool-policy labels and golden trajectories. This repository includes synthetic SGD-style sample data for offline smoke tests.

## Transformation

Dialogue turns are parsed into rows containing dialogue context, user utterance, service, intent, known slots, missing slots, risk flags, and a framework tool label. Train, calibration, and final evaluation splits are kept disjoint. The calibration split is used for confidence-threshold selection; final metrics and golden trajectories use only the held-out evaluation split.

## Sizes

- Train rows: 4616
- Calibration rows: 590
- Final evaluation rows: 571
- Golden trajectories: 571
- Tool labels: ask_clarification, calendar_lookup, calendar_write, media_search, safety_check, search_docs, search_places, weather_lookup

## Claim boundary

This project uses public-data-compatible and synthetic sample inputs. It does not use Apple data, private user data, production Siri data, or real customer workflows.
