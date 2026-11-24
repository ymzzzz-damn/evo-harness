# Evo Harness

A source-only framework for learning parameterized robot skills on RoboCasa through two loops: task exploration and code-library maintenance.

The current catalog defines **32 executable code interfaces** across 13 families: 3 perception interfaces (P01–P03), 26 action interfaces (A01–A26), and 3 coordination interfaces (C01–C03). Every interface uses `skill(api, params) -> dict`, starts without an implementation, and can be generated, repaired, parameterized, versioned, and called by another published skill. A catalog entry is a capability contract, not an implemented or benchmark-validated policy.

The repository contains framework source, tests, static catalogs, deployment utilities, and an example configuration. It contains **no learned code tools, populated skill libraries, model transcripts, simulator assets, credentials, or experiment outputs**.

See [中文说明](README.zh-CN.md) for the complete capability table and detailed usage.

## Two loops

1. **Explore:** use the task instruction, current RGB-D observations, robot measurements, published skills, and relevant failure memories to plan the next atomic step. Reuse or generate code, execute it, and independently check its declared postcondition. Record each attempt. After a verified step, plan the next step from the updated state.
2. **Maintain:** at each episode boundary, process the successful and failed attempts even if the overall task failed. Parameterize successful code for reuse. A previously empty interface requires fresh-environment replay before its first publication. For an already published interface, keep equivalent code or compare changed code and its evidence before retaining, updating, or saving a draft; this comparison path does not repeat replay. Parameterize failed code with observed feedback and explicitly unverified cause hypotheses for later repair.

Failed drafts are not active skills and cannot be called as published dependencies. Comparison-based updates retain `replay_verified=false`; code comparison does not prove regression success. Task completion always uses the native benchmark verdict, independently of skill publication.

The example configuration selects `compare_existing`, one-step planning, attempt memory, and episode-boundary maintenance. The optional `replay` admission policy retains full regression admission, with optional reuse of exact matching successful replay evidence within a session. Episode reset retries are configured separately from attempts within an episode; `episode_retry.max_episodes: null` continues retryable failures until success or an external stop, while preserving the native horizon per episode.

## Perception, coordination, and verification

`api.read_camera(camera)` gives generated code copies of the current RGB/depth arrays and calibration for NumPy processing. It supplies no hidden labels or built-in visual-language recognizer. Model requests receive images separately. Published P/A/C functions can call explicitly declared, version-locked dependencies through `api.call`; returned data can also be passed into later plan parameters. The host retains control of reset, overall budgets, benchmark scoring, and publication.

The host validates physical postconditions and three limited output contracts: `observed_result` compares selected outputs with public observations or depth projection; `effect_report` checks a P03 report against an independent local condition; `assignment_valid` checks C01 assignment constraints conditional on the supplied inputs. These output checks do not establish unrestricted visual recognition accuracy, untested attributes, or physical task completion. Unsupported conditions remain unknown.

The generated-code API exposes low-level motion, camera access, and calls to published skills. It has no prebuilt pick/place/press policies. AST checks and execution deadlines are not an operating-system security sandbox; run generated code in an appropriately controlled worker environment.

## Install and run

Python 3.10 or newer is required.

```bash
git clone https://github.com/ymzzzz-damn/evo-harness.git
cd evo-harness
python3 -m pip install -e '.[test,control]'
python3 -m pytest tests -q
python3 -m evo.cli init --library ./library_pac32
python3 -m evo.cli catalog
cp configs/robocasa.example.json configs/robocasa.local.json
```

Real simulation additionally requires compatible RoboCasa, robosuite, MuJoCo, RoboEnv, `roboenv_runtime`, assets, and rendering dependencies. The package installation above does not install that simulator stack. The adapter uses RoboEnv's low-level backend, without importing its high-level skill implementations.

The example uses a local compatible-server endpoint and a `YOUR_VISION_MODEL` placeholder. Edit `configs/robocasa.local.json` for your model, provider endpoint, tasks, library path, and episode budget. It sets `episode_retry.max_episodes` to `null`; use a positive integer to bound reset retries. Provide the credential through the environment variable named by `agent.api_key_env`; do not put credentials in configuration files. Relative paths resolve from the process working directory. With the simulator and provider configured:

```bash
python3 -m evo.cli run --config configs/robocasa.local.json --output runs/local-001
python3 -m evo.cli inspect --library ./library_pac32
```

Each output directory must be new; the library may be reused across tasks. `explore` and `maintain` can run the loops separately. `evaluate` executes reviewed plans against fixed skills; `evaluate-policy` supports the framework's separate policy-evaluation workflow. Neither substitutes training success for held-out evaluation.

## Source layout

- `evo/atomic_skills.json`: the single source of capability definitions, parameters, descriptions, and contracts.
- `evo/prompts.py`, `evo/agent.py`: model instructions, response contracts, and providers.
- `evo/loops.py`, `evo/library.py`: exploration, maintenance, failure memory, and versioned publication.
- `evo/runtime_api.py`, `evo/robocasa_adapter.py`, `evo/result_verification.py`: restricted runtime API and host verification.
- `configs/human300_capability_requirements.json`: static training-task requirements for offline auditing, not runtime plans or evidence of task success.
- `scripts/`: campaign, evaluation, recovery, and deployment utilities. The P6000 launchers are optional cluster-specific adapters requiring an existing runtime, job template, shared mounts, and Kubernetes configuration; they are not a portable simulator installer.

Tests use deterministic fakes where appropriate. Passing them checks framework behavior and does not establish RoboCasa task success or generalization.
