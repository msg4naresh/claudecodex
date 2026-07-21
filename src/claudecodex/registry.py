"""Central provider registry.

To add a provider: give its module a `PROVIDER = ProviderEntry(...)` (see
bedrock.py for the pattern), then add one import + one entry below.
server.py never needs to change - it only talks to this registry.
"""

from typing import List

from claudecodex.provider import ProviderEntry
from claudecodex.bedrock import PROVIDER as _bedrock
from claudecodex.openai_compatible import PROVIDER as _openai_compatible
from claudecodex.copilot_provider import PROVIDER as _copilot

DEFAULT_PROVIDER = "copilot"

_PROVIDERS = {p.name: p for p in (_bedrock, _openai_compatible, _copilot)}


def provider_names() -> List[str]:
    return sorted(_PROVIDERS)


def get_entry(name: str) -> ProviderEntry:
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: {name}. Must be one of: "
            f"{', '.join(provider_names())}"
        )
    return _PROVIDERS[name]
