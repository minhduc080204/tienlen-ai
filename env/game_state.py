# env/game_state.py
from dataclasses import dataclass, field
from core.card import Card


@dataclass
class GameState:
    hands: list[list[Card]]
    current_player: int
    current_trick: list[Card] | None = None
    last_player: int | None = None
    passed_players: list[int] = field(default_factory=list)  # Player IDs who passed in current round
    finished: bool = False
    winner: int | None = None
    discard_pile: list[Card] = field(default_factory=list)
