import random

import pytest

from kuhn_env_explore import POLICIES, J, K, Q, make_env_factory, payoff_check, play_match, run_matchups

LEGAL_STATES = [{"check", "bet"}, {"fold", "call"}]


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_policies_only_return_legal_actions(name):
    rng = random.Random(0)
    for card in (J, Q, K):
        for first in (True, False):
            for legal in LEGAL_STATES:
                for _ in range(50):
                    action = POLICIES[name](card, first, legal, rng)
                    assert action.strip("[]") in legal


def test_equilibrium_never_bluff_calls_with_jack():
    rng = random.Random(0)
    for first in (True, False):
        assert all(POLICIES["equilibrium"](J, first, {"fold", "call"}, rng) == "[fold]" for _ in range(100))


textarena = pytest.importorskip("textarena")


def test_called_bet_does_not_change_stakes_in_textarena():
    assert payoff_check(make_env_factory("textarena"))["called_bet_changes_stakes"] is False


def test_called_bet_changes_stakes_with_fix():
    assert payoff_check(make_env_factory("textarena_fixed"))["called_bet_changes_stakes"] is True


def test_match_is_deterministic_for_a_seed():
    make_env = make_env_factory("textarena")
    a = play_match(make_env, POLICIES["random"], POLICIES["equilibrium"], seed=11)
    b = play_match(make_env, POLICIES["random"], POLICIES["equilibrium"], seed=11)
    assert a == b


def test_three_hand_match_has_three_hand_results():
    result = play_match(make_env_factory("textarena"), POLICIES["never_fold"], POLICIES["never_fold"], seed=0)
    assert len(result.hand_winners) == 3
    assert 6 <= result.decisions <= 9


def test_run_matchups_report_shape():
    report = run_matchups(make_env_factory("textarena"), n_games=20)
    for m in report["matchups"].values():
        assert m["a_win"] + m["a_draw"] + m["a_loss"] == pytest.approx(1.0)
    assert report["winner_rule"]["agrees_with_chip_leader"] == 1.0
