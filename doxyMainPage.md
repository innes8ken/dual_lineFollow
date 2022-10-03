
/**\mainpage
 *
 * \section intro_sec FCL and BCL Linefollower
 *
  Forward error propagation within a closed loop system has been theorised and studied using a line following robot simulation (Porr & Miller, 2019). This project aims to incorporate the forward error propagation paradigm into a physical line-following robot serving as the autonomous agent. 
  The robot’s ability to learn using the predictive signals from a camera array will be analysed and compared to the same robot running a 
  backpropagation algorithm (Daryanavard & Porr, n.d.)
 
  [Github Page dual_linefollower](https://github.com/innes8ken/dual_lineFollow)

# Feedforward Closedloop Learning (FCL)

[Forward propagation closed loop learning
Bernd Porr, Paul Miller. Adaptive Behaviour 2019.](https://www.berndporr.me.uk/Porr_Miller_FCL_2019_Adaptive_Behaviour.pdf)

For an autonomous agent, the inputs are the sensory data that inform the agent of the state of the world, and the outputs are their actions, which act on the world and consequently produce new sensory inputs. The agent only knows of its own actions via their effect on future inputs; therefore desired states, and error signals, are most naturally defined in terms of the inputs. Most machine learning algorithms, however, operate in terms of desired outputs. For example, backpropagation takes target output values and propagates the corresponding error backwards through the network in order to change the weights. In closed loop settings, it is far more obvious how to define desired sensory inputs than desired actions, however. To train a deep network using errors defined in the input space would call for an algorithm that can propagate those errors forwards through the network, from input layer to output layer, in much the same way that activations are propagated.

[Github project page](https://github.com/glasgowneuro/feedforward_closedloop_learning)

# Backpropagation Closedloop Learning (BCL)

Traditional backpropagation paradigms can learn by comparing the
output of the system and then propagate the error in the opposite direction to neural
activation. This method can successfully be implemented into a closed-loop system for
deep learning (Daryanavard & Porr, n.d.)

[Daryanavard, S. and Porr, B. (2020) Closed-loop deep learning: generating forward models with backpropagation](https://eprints.gla.ac.uk/226317/)
