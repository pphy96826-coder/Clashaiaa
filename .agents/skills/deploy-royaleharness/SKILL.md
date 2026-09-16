---
name: deploy-royaleharness
description: Deploy this modified RoyaleHarness on Apple Silicon macOS, configure separate online and optional offline MuMu instances, build probes, validate dependencies and diagnose deployment failures. Use for installing or migrating this project, not model training or general game strategy.
---

# Deploy RoyaleHarness

Locate the checkout containing main.py, deployment/ and config.py. Read docs/PROJECT_GUIDE.zh-CN.md from that checkout before deployment. It documents the current Mac fork; the older Windows configuration skill is not a Mac recipe.

1. Inventory Python architecture, Tk, Torch backend, FirstLight directory/catalogs/checkpoints, ADB devices, game library version and saved configuration. Use CR_AGENT_SETTINGS consistently for GUI and CLI. Never infer the online serial from the offline one or reuse a developer account ID. Existing settings.device2.local.json can override the GUI default.
2. Install dependencies from deployment/requirements-macos-arm64.lock in a new venv. If a pinned distribution is unavailable, identify it; don't silently upgrade and claim reproducibility. Review upstream.lock.json as a historical reference, not proof that the local modified FirstLight checkout is identical.
3. Confirm online game SHA/ABI, compile with deployment/build_live_probe.sh and use tools/install_probe.py with the explicit stale-recovery candidate path. Installation backs up the original SDK and restarts the selected game. Preserve backup identity; never replace the real SDK with another proxy.
4. Confirm GET is valid JSON, then observe real advancing ticks in a battle; lobby false is normal. Read account identity from tools/inspect_players.py, verify screen coordinates and the independent hero button. Compile the macOS lifecycle OCR helper if continuous mode is requested.
5. Run preflight, then --dry-run --once without matchmaking flags. Perform real input or continuous matches only within the user's requested deployment/testing scope. Never replay an ambiguous previous touch. Distinguish transport completion, hand/skill confirmation and deployment.
6. Offline engine is optional and uses a separate serial and port. Installing its APK on the online serial overwrites the online app. Require an explicit distinct offline target. Do not treat ready as proof of working replay or model integration. Validate configure, card consume, snapshot/restore and measured rollout time. Partial-opponent mirror is disabled for model input; report that limitation.
7. Report exact files/config locations, version fingerprints, completed validation levels and unresolved dependencies. Do not expose full account configuration, SDK backups, raw logs, APKs or weights in a public upload.

For release packaging use deployment/package_portable.py; its manifest excludes runtime files. Third-party binary redistribution requires its own license review. The skill does not authorize uploading to the upstream author's repository.
