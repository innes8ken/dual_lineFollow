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
 * This is where internal parameters for the BCL learning paradigm are set 
 * 
 * @param numInputLayers number of predictive input layers from camera array before filtering 
 * @param sampleRate sampling rate of network? 
 */
void initialize_samanet(int numInputs_pi, double sampleRate = 30.f);




/**
 * @brief Running the NN on each BCL iteration 
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

/**
 * @brief initialising the FCL paradigm 
 * 
 * This will take all internal parameters for the FCL paradigm are set 
 * 
 * @param num_of_inputs Number of inputs from the predictive sensor
 * @param num_of_neurons_per_layer_array  Array containing the desired number of neurons per layer
 * @param num_layers Number of layers - must match the array above!
 * @param num_filtersInput Number of input filters. Will be 5 to match the BCL algorithm 
 * @param minT Minimum temporal delay for the filtered predictive signals 
 * @param maxT Maximum temporal delay from that filtered predictive signals 
 */
void initialize_fclNet(int num_of_predictors); //, int* num_of_neurons_per_layer_array, int num_layers, int num_filtersInput, double minT, double maxT);

/**
 * @brief Running the nn on each FCL iteration
 * 
 * @param in predictor differences 
 * @param error reflex error 
 * @return double 
 */
double run_fclNet(std::vector<double> &in, double error);