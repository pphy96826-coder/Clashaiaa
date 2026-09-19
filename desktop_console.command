#!/bin/zsh

set -e
cd "$(dirname "$0")"

if [[ -z "${CR_AGENT_PYTHON:-}" && -x "$HOME/Documents/Codex/RoyaleHarness/.venv/bin/python" ]]; then
  export CR_AGENT_PYTHON="$HOME/Documents/Codex/RoyaleHarness/.venv/bin/python"
fi
if [[ -z "${CR_AGENT_SETTINGS:-}" && -f "$HOME/Documents/Codex/RoyaleHarness/settings.local.json" ]]; then
  export CR_AGENT_SETTINGS="$HOME/Documents/Codex/RoyaleHarness/settings.local.json"
fi

console_python="${CR_CONSOLE_PYTHON:-${CR_AGENT_PYTHON:-}}"
if [[ -z "$console_python" || ! -x "$console_python" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    console_python=".venv/bin/python"
  else
    console_python="$(command -v python3)"
  fi
fi

exec "$console_python" desktop_console.py
