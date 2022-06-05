# ReinforcementLearningBasedPacMan

In Pac-Man, the player tries to finish the yellow discs by moving through a maze. The player, whose goal is to collect all the small discs by avoiding ghosts and monsters, moves to the next stage when he collects all the discs. Collecting the fruits that appear on the maze gives the player extra points. When he receives the large yellow discs, monsters and ghosts turn blue and become edible for a while.

### Milestones of Project

We describe our deep Q-learning neural network. This is a CNN network that takes in-game screenshots and outputs the probabilities of each action or Q-value in the Pac-Man playground. To obtain a probability tensor, we do not include any activation functions in our last layer.
Since Q-learning requires us to have knowledge of both current and future states, we need to start with data generation. We feed the pre-processed input images of the playground representing the initial states into the network and obtain the initial probability distribution of the actions or Q-values. Before training, these values will appear randomly and as non-optimal.
Using the argmax() function with our probability tensor, we select the action with the highest probability available and use that to create an epsilon greedy policy.
Using our policy, we will select action a and evaluate our decision to receive information about new states in the gym environment model, reward r, and whether the section has been completed.
We store this combination of information in a buffer in the form of a list *<s,a,r,s', d>* and repeat steps 2–4 a predetermined number of times to create a sufficiently large buffer dataset.
Once step 5 is complete, we move on to create our target y-values,* R'* and *A'*, required for the loss calculation. We get A' by feeding S into our network, even though the former is discounted from R.
With all its components in place, we then calculate the loss of the trained network.
After the training is over, we will evaluate the performance of our agent graphically and through demonstration.
