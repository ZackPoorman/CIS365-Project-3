import retro        #pip install gym-retro
import numpy as np  #pip install numpy
import cv2          #pip install opencv-python==4.1.2.30
import neat         #pip install neat-python
import pickle       #pip install pickle

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
    
    def work(self):
        #Swap BalloonFight with Arkanoid and vice versa
        self.env = retro.make('Arkanoid-Nes', 'Level1')
        self.env.reset()

        ob, _, _, _ = self.env.step(self.env.action_space.sample())
        inx = int(ob.shape[0]/8)
        iny = int(ob.shape[1]/8)

        fitness = 0
        done = False
        ann = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        screen_input = []

        cv2.namedWindow("Main", cv2.WINDOW_NORMAL)

        while not done:
            #Uncomment for hilarity
            #self.env.render()

            #Displays the games as AI sees it but with color
            screen_scaled = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
            screen_scaled = cv2.resize(screen_scaled, (iny, inx))

            #Resize the screen input for faster computation
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            cv2.imshow('main', screen_scaled)
            cv2.waitKey(1)

            #Flatten screen into single row of input
            screen_input = np.ndarray.flatten(ob)

            actions = ann.activate(screen_input)
            ob, rew, done, info = self.env.step(actions)

            #Evaluating Genome
            fitness += rew
        print(fitness)    
        return fitness

def evaluate_genomes(genome, config):

    worker = Worker(genome, config)
    return worker.work()


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

#Set the first parameter to number of cores to use
pe = neat.ParallelEvaluator(2, evaluate_genomes)

winner = p.run(pe.evaluate)

#Saves current ANN
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)