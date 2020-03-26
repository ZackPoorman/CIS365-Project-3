import retro        #pip install gym-retro
import numpy as np  #pip install numpy
import cv2          #pip install opencv-python==4.1.2.30
import neat         #pip install neat-python
import pickle       #pip install pickle

#Swap BalloonFight with Arkanoid and vice versa
env = retro.make('BalloonFight-Nes', 'Level1')
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


        #Uncomment to show what ANN sees
        #cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        while not done:
            #comment out to show what ANN sees
            env.render()
            
            frame += 1
            #Scaledimg is used for showing ANN visually
            scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            scaledimg = cv2.resize(scaledimg, (iny, inx))

            #Actually resizes the screenshot input
            ob = cv2.resize(ob, (inx, iny))
            #Turn screenshot input -> greyscale (simplifies input)
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            #Resize the input for the ANN
            ob = np.reshape(ob, (inx, iny))

            #Uncomment to show what ANN sees
            #cv2.imshow('main', scaledimg)
            #cv2.waitKey(1)
            
            #Flatten screen for input
            imgarray = np.ndarray.flatten(ob)
            
            nnOutput = net.activate(imgarray)

            ob, rew, done, info = env.step(nnOutput)
            #imgarray.clear()

            score = info['score']

            #For a basic neat alg comment this if statement
            if score > score_max:
                fitness_current += 1
                score_max = score
            
            #Uncomment this for the basic neat alg
            #fitness_current += rew

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
                fitness += 1
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

#Creates the population based on the config file
p = neat.Population(config)

#Creates statistics for each generation
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

#Saves current ANN
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
