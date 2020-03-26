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
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        imgarray = []

        cv2.namedWindow("Main", cv2.WINDOW_NORMAL)

        while not done:
            #Uncomment for hilarity
            #self.env.render()

            #Displays the games as AI sees it but with color
            scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
            scaledimg = cv2.resize(scaledimg, (iny, inx))

            #Resize the screen input for faster computation
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            cv2.imshow('main', scaledimg)
            cv2.waitKey(1)

            #Flatten screen into single row of input
            imgarray = np.ndarray.flatten(ob)

            actions = net.activate(imgarray)
            ob, rew, done, info = self.env.step(actions)

            #Evaluating Genome
            fitness += rew
        print(fitness)    
        return fitness

def eval_genomes(genome, config):

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
pe = neat.ParallelEvaluator(2, eval_genomes)

winner = p.run(pe.evaluate)

#Saves current ANN
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)