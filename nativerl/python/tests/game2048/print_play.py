import logic
from base import Game2048


def render(matrix):
    for row in matrix:
        print(row)
    print(f"Total reward: {game.total_reward}, steps: {game.steps}")


game = Game2048(random_movements=True, human=True)

while not game.is_done():
    render(game.matrix)
    print(game.get_observation())
    if not game.random:
        game.action = input("\nMake a move: Up: 0, Down: 1, Left: 2, Right: 3\n")
    game.step()

if logic.game_state(game.matrix) == "win":
    print("You win!")
else:
    print("You lose :(")
print(f"Total reward: {game.total_reward}, steps: {game.steps}")
