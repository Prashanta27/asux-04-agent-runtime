# AsuX Agent Runtime V1

Production-oriented single AI Agent Runtime for the AsuX AI Workforce platform.

## Architecture Overview
The Agent Runtime orchestrates the execution loop for a Single AI Commerce Assistant, consuming:
1. `asux-llm-gateway` (Project 1) for unified model access.
2. `asux-prompt-engine` (Project 2) for centralized, versioned prompt definitions and rendering.
3. `asux-tool-runtime` (Project 3) for tool discovery, permission authorization, schema validation, and execution.

See `docs/architecture.md` for details.
