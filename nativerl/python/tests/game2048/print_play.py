from base import Game2048
import logic


def render(matrix):
    for row in matrix:
        print(row)
    print(f'{game.total_reward=} {game.steps=} {game.rew=}')


game = Game2048(random_movements=True, human=True)

while not game.is_done():
    render(game.matrix)
    print(game.get_observation())
    if not game.random:
        game.action = input("\nMake a move: Up: 0, Down: 1, Left: 2, Right: 3\n")
    game.step()

if logic.game_state(game.matrix) == 'win':
    print("You win!")
else:
    print("You lose :(")
print(f'{game.total_reward=} {game.steps=} {game.rew=}')