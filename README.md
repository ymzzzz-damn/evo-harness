# Evo Harness

A source-only framework for learning parameterized atomic robot skills on RoboCasa.

The exploration loop plans tasks, generates and repairs Python policies, and saves independently verified atomic successes. The maintenance loop replays new candidates in fresh environments, checks established regression cases, and publishes versioned skills. An atomic skill can be published even when a later task step fails.

The repository includes the P01–P40 catalog, the empty-library implementation, RoboCasa adapter, model providers, CLI, deployment utilities, tests, and a configuration template. It contains **no learned code tools, populated skill libraries, model transcripts, simulator assets, or experiment outputs**.

See [中文说明](README.zh-CN.md) for architecture, setup, configuration, and usage.

```bash
python3 -m pip install -e '.[test,control]'
python3 -m pytest tests -q
python3 -m evo.cli init --library ./library
```

Real simulation additionally requires compatible RoboCasa, robosuite, MuJoCo, RoboEnv, `roboenv_runtime`, assets, and rendering dependencies. These are not installed by the commands above. Configure your vision model using `configs/robocasa.example.json`; keep credentials in environment variables.
