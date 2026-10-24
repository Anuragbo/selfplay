"""SPIRAL's KuhnPoker-v1 rules on TextArena 0.7.4.

A port of third_party/spiral_kuhn_env.py (written for textarena==0.6.4) to the 0.7.4 State API that
UnstableBaselines installs. Game logic, prompt and message text follow SPIRAL exactly, including its quirks,
so that training on this environment is a faithful replication:

- 5 hands per match; the first hand is opened by player 1 and the opener alternates.
- Each player antes 1 chip; [bet] and [call] add nothing to the pot, so every hand is worth +-1 chip.
- The winner of the final hand wins the match (rewards +1/-1), whatever the chip totals. No draws.
- One invalid move per turn is forgiven (the player retries); a second ends the match with rewards -1 for the
  offender and 0 for the opponent (0.7.4's own TwoPlayerState would give the opponent +1).
- The next hand is dealt lazily in get_observation(), as in SPIRAL, so the deck shuffles consume the global
  random stream in the same order. With the same seed, deals match SPIRAL's environment exactly.
"""

import random
import re
from typing import Any

import textarena as ta

ACTION_PATTERN = re.compile(r"\[(Check|Bet|Fold|Call)\]", re.IGNORECASE)
LEGAL_ACTION_TREE = {
    "check": {"check": "showdown", "bet": {"fold": "loser", "call": "showdown"}},
    "bet": {"fold": "loser", "call": "showdown"},
}
RANKS = {0: "J", 1: "Q", 2: "K"}

_GAME = ta.ObservationType.GAME_MESSAGE
_ACTION = ta.ObservationType.PLAYER_ACTION
_ADMIN = ta.ObservationType.GAME_ADMIN


class SpiralKuhnPokerEnv(ta.Env):
    def __init__(self, ante: int = 1, max_rounds: int = 5):
        super().__init__()
        self.ante = ante
        self.max_rounds = max_rounds
        self.deck = [0, 1, 2]  # 0=J, 1=Q, 2=K

    def reset(self, num_players: int, seed: int | None = None):
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed, max_turns=self.max_rounds)
        game_state = {"pot": None, "player_chips": {0: 0, 1: 0}, "current_round": 0, "starting_player": 0}
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)
        self._init_round()

    def get_observation(self):
        if self.state.game_state.get("round_ended", False):
            self.state.game_state["round_ended"] = False
            self._init_round()
        return self.state.current_player_id, self.state.get_current_player_observation()

    def _init_round(self):
        gs = self.state.game_state
        random.shuffle(self.deck)
        gs["player_cards"] = {0: self.deck[0], 1: self.deck[1]}
        gs["pot"] = self.ante * 2
        gs["player_chips"][0] -= self.ante
        gs["player_chips"][1] -= self.ante
        gs["current_round"] += 1
        gs["current_legal_action_tree"] = LEGAL_ACTION_TREE.copy()

        starting_player = 1 - gs["starting_player"]
        gs["starting_player"] = starting_player
        # Assigned directly, as in SPIRAL: a forgiven invalid move is not reset at a new hand.
        self.state.current_player_id = starting_player

        for player_id in range(2):
            message = (
                f"Starting round {gs['current_round']} out of {self.max_rounds} rounds.\n"
                f"Your card is: {RANKS[gs['player_cards'][player_id]]}\n"
            )
            if player_id == starting_player:
                message += f"Your available actions are: {self._legal_actions_str()}"
            self.state.add_observation(message=message, observation_type=_GAME, to_id=player_id)

    def _prompt(self, player_id: int, game_state: dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in a {self.max_rounds} round game of Kuhn Poker.\n"
            f"Game Rules:\n"
            f"- Kuhn Poker uses a 3-card deck with J, Q, K (J lowest, K highest)\n"
            f"- Each player antes {self.ante} chip and receives 1 card each round\n"
            f"- Game continues for {self.max_rounds} rounds\n"
            f"- The player with the most chips after all rounds wins\n\n"
            f"Action Rules:\n"
            f"- '[check]': Pass without betting (only if no bet is on the table)\n"
            f"- '[bet]': Add 1 chip to the pot (only if no bet is on the table)\n"
            f"- '[call]': Match an opponent's bet by adding 1 chip to the pot\n"
            f"- '[fold]': Surrender your hand and let your opponent win the pot\n\n"
        )

    def _legal_actions_str(self) -> str:
        return ", ".join(f"[{k}]" for k in self.state.game_state["current_legal_action_tree"])

    def step(self, action: str) -> tuple[bool, dict[str, Any]]:
        if self.state.done:
            return True, self.state.step_info
        player_id = self.state.current_player_id
        self.state.add_observation(message=action, observation_type=_ACTION, from_id=player_id)

        match = ACTION_PATTERN.search(action.strip())
        if not match:
            return self._invalid_move("Action must be [Check], [Bet], [Call], or [Fold].")
        move = match.group(1).lower()
        if move not in self.state.game_state["current_legal_action_tree"]:
            return self._invalid_move(f"Action must be {self._legal_actions_str()}.")

        tree = self.state.game_state["current_legal_action_tree"][move]
        self.state.game_state["current_legal_action_tree"] = tree
        if tree == "loser":
            self._set_round_winner(1 - player_id, f"Player {player_id} has folded.")
            return self.state.done, self.state.step_info
        if tree == "showdown":
            self._handle_showdown()
            return self.state.done, self.state.step_info

        self.state.add_observation(
            message=f"Your available actions are: {self._legal_actions_str()}", observation_type=_GAME,
            to_id=1 - player_id,
        )
        return self.state.step()

    def _invalid_move(self, reason: str) -> tuple[bool, dict[str, Any]]:
        state, player_id = self.state, self.state.current_player_id
        if state.error_allowance > state.error_count:
            state.error_count += 1
            state.made_invalid_move = True
            state.add_observation(
                message=(
                    f"Player {player_id} attempted an invalid move. Reason: {reason} "
                    f"Please resubmit a valid move and remember to follow the game rules to avoid penalties."
                ),
                observation_type=_ADMIN,
            )
        else:
            state.rewards = {pid: (-1 if pid == player_id else 0) for pid in range(2)}
            state.add_observation(
                message=f"Player {player_id} lost the game by way of invalid move. Reason: {reason}",
                observation_type=_ADMIN,
            )
            state.game_info[player_id]["invalid_move"] = True
            for pid in range(2):
                state.game_info[pid]["reason"] = f"Invalid Move: {reason}"
            state.done = True
        return state.step()

    def _set_round_winner(self, player_id: int, reason: str):
        gs = self.state.game_state
        gs["player_chips"][player_id] += gs["pot"]
        self.state.add_observation(message=reason, observation_type=_GAME)
        if gs["current_round"] >= self.max_rounds:
            reason = f"Player {player_id} won by having more chips at the end of all {self.max_rounds} rounds."
            self.state.add_observation(message=f"Player {player_id} won the game. Reason: {reason}",
                                       observation_type=_ADMIN)
            # TwoPlayerState.set_winners is broken in 0.7.4 (it writes to a missing `info` attribute).
            self.state.set_winner(player_id=player_id, reason=reason)
        else:
            gs["round_ended"] = True

    def _handle_showdown(self):
        card_p0, card_p1 = self.state.game_state["player_cards"][0], self.state.game_state["player_cards"][1]
        self.state.add_observation(
            message=f"Cards: Player 0 had {RANKS[card_p0]}, Player 1 had {RANKS[card_p1]}", observation_type=_GAME
        )
        winner = 0 if card_p0 > card_p1 else 1
        winner_card, loser_card = (card_p0, card_p1) if winner == 0 else (card_p1, card_p0)
        reason = (
            f"Showdown: Player {winner}'s {RANKS[winner_card]} beats "
            f"Player {1 - winner}'s {RANKS[loser_card]}. "
            f"Player {winner} wins pot of {self.state.game_state['pot']} chips."
        )
        self._set_round_winner(winner, reason)
