import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make('Arkanoid-Nes', 'Level1')
imgarray = []
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:

        #image of screen @ time of action
        ob = env.reset()
        #action of agent
        ac = env.action_space.sample()
    
        #inputs: x, y (size of screen), colors
        inx, iny, inc = env.observation_space.shape

        #Scale the image down to make the learning faster
        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        score = 0
        score_max = 0
        done = False

        while not done:
            env.render()
            frame += 1

            #Actually resizes the screenshot input
            ob = cv2.resize(ob, (inx, iny))
            #Turn screenshot input -> greyscale (simplifies input)
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            #Resize the input for the ANN
            ob = np.reshape(ob, (inx, iny))

            for x in ob:
                for y in x:
                    imgarray.append(y)
            
            nnOutput = net.activate(imgarray)

            ob, rew, done, info = env.step(nnOutput)
            imgarray.clear()

            score = info['score']

            if score > score_max:
                fitness_current += 1
                score_max = score

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)

winner = p.run(eval_genomes)