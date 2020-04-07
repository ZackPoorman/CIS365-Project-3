import retro        #pip install gym-retro
import numpy as np  #pip install numpy
import cv2          #pip install opencv-python==4.1.2.30
import neat         #pip install neat-python
import pickle       #pip install pickle

#IMPORTANT! You need the ROMS of Balloon Fight and Arkanoid to run this.

#Note the original config-feedforward file was taken from https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT and we tweaked it for our needs.
#We used the Sonic tutorial as a base to familiarize ourselves with. 
#We then implemented the NEAT algorithm based on the OpenAI-Gym-Retro features for Balloon Fight (score and pixel of water).
#To run: ./TrainingAlg.py

#Swap BalloonFight with Arkanoid and vice versa
env = retro.make('BalloonFight-Nes', 'Level1')
screen_input = []
def evaluate_genomes(genomes, config):
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

        ann = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        curr_max_fitness = 0
        fitness = 0
        frame = 0
        incr = 0
        score = 0
        score_max = 0
        done = False

        #We have to adjust for level changes
        prevScreenInput = None
        noScreenChange = False
        #Uncomment to show what ANN sees
        #cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        while not done:
            #comment out to hide what ANN sees
            env.render()
            
            """
            The bottom of the main stages only contains water; use the
            water pixel to tell if we're changing levels OR on a bonus stage or not.
            """
            newLevel = (np.array_equal(ob[len(ob)-1][len(ob)-1],[0,0,168]))
            
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
            screen_input = np.ndarray.flatten(ob)
            
            noScreenChange = (np.array_equal(prevScreenInput,screen_input))
                
            prevScreenInput = screen_input
            
            nnOutput = ann.activate(screen_input)

            ob, rew, done, info = env.step(nnOutput)
            #screen_input.clear()

            score = info['score']

            #For a basic neat alg comment this if statement
            if score > score_max:
                fitness += 1
                score_max = score
            
            #Uncomment this for the basic neat alg
            #fitness += rew

            #Reset done counter if fitness is increased from previous best
            if fitness > curr_max_fitness:
                curr_max_fitness = fitness
                incr = 0
            #if it's a bonus stage, reset the done counter
            elif not newLevel:
                incr = 0
            #if there was not a screen change, increment the done counter
            elif not noScreenChange:
                incr += 1

            if done or incr == int(500 + frame / 50):
                done = True
                print("Genome ID: " + str(genome_id) + " Fitness: " + str(fitness) + " Frames: "+str(frame))

            genome.fitness = fitness

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

#Creates the population based on the config file
p = neat.Population(config)

#insert desired filename to load a checkpoint
p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-78")

#Creates statistics for each generation
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10,10000))

winner = p.run(evaluate_genomes)

#Saves current ANN
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
