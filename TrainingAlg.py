import retro

def main():
    env = retro.make(game = 'Game Name')
    obs = env.reset()