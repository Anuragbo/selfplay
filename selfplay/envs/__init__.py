"""Game environments, registered with TextArena on request (no work at import time)."""

SPIRAL_KUHN_ID = "SpiralKuhnPoker-v1"


def register_envs() -> None:
    """Register our environments with TextArena's registry. Safe to call more than once."""
    from textarena.envs import BOARDGAME_WRAPPERS, DEFAULT_WRAPPERS
    from textarena.envs.registration import ENV_REGISTRY, register_with_versions

    if SPIRAL_KUHN_ID not in ENV_REGISTRY:
        register_with_versions(
            id=SPIRAL_KUHN_ID,
            entry_point="selfplay.envs.kuhn_spiral:SpiralKuhnPokerEnv",
            wrappers={"default": DEFAULT_WRAPPERS, "-train": BOARDGAME_WRAPPERS},
            ante=1,
            max_rounds=5,
        )
