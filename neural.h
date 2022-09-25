#pragma once
#include <vector>

namespace cv {

/**
 * @brief 
 * 
 */
class Mat;
}

/**
 * @brief initialising the NN 
 * 
 * This is where internal parameters for the learning paradigm are set 
 * 
 * @param numInputLayers number of predictive inputs from camera array before filtering 
 * @param sampleRate sampling rate of network? 
 */
void initialize_samanet(int numInputLayers, double sampleRate = 30.f);

/**
 * @brief Running the NN on each iteration
 * 
 * Returns the overall output of the NN
 * 
 * @param in Predictor deltas
 * @param error 
 * @return double 
 */
double run_samanet(std::vector<double> &in, double error);

/**
 * @brief 
 * 
 */
void save_samanet();
