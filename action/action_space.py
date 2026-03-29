# # action/action_space.py

from action.action_spec import ActionSpec
from core.move_type import MoveType

ACTION_SPACE = []

# PASS
ACTION_SPACE.append(ActionSpec(MoveType.PASS, 0))

# SINGLE
ACTION_SPACE.append(ActionSpec(MoveType.SINGLE, 1))

# PAIR
ACTION_SPACE.append(ActionSpec(MoveType.PAIR, 2))

# TRIPLE
ACTION_SPACE.append(ActionSpec(MoveType.TRIPLE, 3))

# FOUR OF A KIND
ACTION_SPACE.append(ActionSpec(MoveType.FOUR_OF_KIND, 4))

# STRAIGHT (tối thiểu 3)
for length in range(3, 12):
    ACTION_SPACE.append(ActionSpec(MoveType.STRAIGHT, length))

# DOUBLE STRAIGHT (6, 8, 10)
for length in [6, 8, 10]:
    ACTION_SPACE.append(ActionSpec(MoveType.DOUBLE_STRAIGHT, length))
