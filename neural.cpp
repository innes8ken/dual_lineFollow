#include "neural.h"
#include "clbp/Net.h"
#include "cvui.h"
#include "bandpass.h"
#include <chrono>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <numeric>
#include <boost/circular_buffer.hpp>
#include <math.h>
#include <ctgmath>
#include "FCL/fcl.h"
#include "FCL/fcl_util.h"

using namespace std;

// setting up buffers for predictor filters
// These filters to ensure that the predictors are delayed in time
// This delay ensures that they correlate with the reflex error (the feedback)
// as the error takes place later after the NN has taken motor actions

std::vector<std::array<Bandpass, 5>> bandpassFilters;
const int numPred = 48;
boost::circular_buffer<double> predVector1[numPred];
boost::circular_buffer<double> predVector2[numPred];
boost::circular_buffer<double> predVector3[numPred];
boost::circular_buffer<double> predVector4[numPred];
boost::circular_buffer<double> predVector5[numPred];

//########################################### Learning Rate Setup #######################################################

double learningExpBCL = -1; // This is the exponential of the leaning rate for the BCL algo 
double lrCoeffBCL = 2; //additional learning rate coefficient (for lrCoeff*10^(learningExp)) for the BCL algo 

double learningExpFCL = -5; // This is the exponential of the leaning rate for the FCL algo 
double lrCoeffFCL = 1; //additional learning rate coefficient (for lrCoeff*10^(learningExp)) for the FCL algo #


//########################################## Declaring BCL global variables ###########################################

// initialising a pointer instance of BCL NN called 'samanet'
std::unique_ptr<Net> samanet;

const int numLayersBCL = 11; // number of layers in the BCL algo 


//########################################## Declaring FCL global variables ###########################################

//initialising a pointer instance of fcl_util class called 'fclFB' (copying samanet)
//std::unique_ptr<FeedforwardClosedloopLearningWithFilterbank> fclFB;

FeedforwardClosedloopLearningWithFilterbank* fclFB = NULL; // initialising fcl nn from class ffcllwf 

static constexpr int numLayersFCL = 6; // number of layers in the FCL algo 
int nNeuronsInLayers[numLayersFCL] = {9,6,6,6,6,6}; // The number of neurons in every layer array
double* reflex_errorArray = NULL;
double* fclInputsArray = NULL;

//long step = 0; 
//double avgError = 0;
//double* pred = NULL;
//double* err = NULL;
 
FILE* flog = NULL;
FILE* llog = NULL;
FILE* fcoord = NULL;

// ########################### initialising the filters used for predictor data ############################################

static void initialize_filters(int numInputs, double sampleRate) {
  // number of inputs are 5* the number of predictors becuase
  // the predictors are filtered by 5 different FIR filters each
  int nPred = numInputs / 5;
  for (int i = 0; i < nPred; i++){
    int j= (int)(i / 6);
    predVector1[i].rresize(1);
    predVector2[i].rresize(2);
    predVector3[i].rresize(3);
    predVector4[i].rresize(4);
    predVector5[i].rresize(5);
  }
  bandpassFilters.resize(numInputs);
  double fs = 1;
  int minT = 100; // minimum time step, the minimum delay
  int maxT = 150; // maximum delay. the other 3 filters will have delays in between Min Max
  double fmin = fs / maxT;
  double fmax = fs / minT;
  double df = (fmax - fmin) / 4.0; // 4 is number of filters minus 1
  for (auto &bank : bandpassFilters) {
    double f = fmin;
    for (auto &filt : bank) {
      filt.setParameters(f, 0.51);
      f += df;
      for(int k=0;k<maxT;k++){
        double a = 0;
        if (k==minT){
          a = 1;
        }
        double b = filt.filter(a);
        assert(b != NAN);
        assert(b != INFINITY);
      }
      filt.reset();
    }
  }
}


void initialize_samanet(int numInputs_Pi, double sampleRate) {
  
  int numInputs_Ui = numInputs_Pi * 5; // 5 is the number of filters                               
  int numNeurons[numLayersBCL]= {};
  int firstLayer = 11; // number of neurons in the first layer
  int lastHiddenLayer = 4; // number of neurons in the last HIDDEN layer
  int incrementLayer = 1;
  int totalNeurons = 0; 
  numNeurons[numLayersBCL - 1] = 3; // number of neurons in the last layer
  for (int i = numLayersBCL - 2; i >= 0; i--){
    numNeurons[i] = lastHiddenLayer + (numLayersBCL - 2 - i)  * incrementLayer;
    totalNeurons += numNeurons[i];
    assert(numNeurons[i] > 0);
  }

  // setting up the BCL NN with the number of layers, neurons per layer and number of inputs
  samanet = std::make_unique<Net>(numLayersBCL, numNeurons, numInputs_Ui);
  // initialising the BCL NN with random weights and no biases and with sigmoid function for activation
  samanet->initNetwork(Neuron::W_RANDOM, Neuron::B_NONE, Neuron::Act_Sigmoid);
  
  
  //double learningRateBCL = exp(learningExp); 
  double learningRateBCL = lrCoeffBCL*(pow(10.0,learningExpBCL));
  
  // printing the learning rate
   
    
  cout << "myLearningRate: " << learningRateBCL << endl;
  samanet->setLearningRate(learningRateBCL); // setting the learning rate
  initialize_filters(numInputs_Ui, sampleRate); // calls the above function to set up the filters
}



void initialize_fclNet(int numInputs_Pi){

  //number of network inputs
  const int nInputs = numInputs_Pi;
	// We set nFilters in the input
	const int nFiltersInput = 10;
	// We set nFilters in the unit
	const int nFilters = 0;
	// Filterbank temporal settings - matching BCL lowpass filter settings 
	const double minT = 100;
	const double maxT = 150;
  // Setting the learning rate 
	double learningRateFCL = lrCoeffFCL*(pow(10.0,learningExpFCL));   //0.00001;
 

 fclFB = new FeedforwardClosedloopLearningWithFilterbank(
  nInputs, 
  nNeuronsInLayers, 
  numLayersFCL, 
  nFiltersInput, 
  minT, 
  maxT);
 
  
  cout << "myLearningRate: " << learningRateFCL << endl;
  //Using fcl_util which has built in filters

  //fclFB = std::make_unique<FeedforwardClosedloopLearningWithFilterbank>(nInputs, nNeuronsInLayers, numLayersFCL, nFiltersInput, minT, maxT);
 fclFB->initWeights(1,0,FCLNeuron::MAX_OUTPUT_RANDOM);
 fclFB->setLearningRate(learningRateFCL);
 fclFB->setLearningRateDiscountFactor(1);
 fclFB->setBias(1);
 fclFB->setActivationFunction(FCLNeuron::TANH);
 fclFB->setMomentum(0.9);		 

} 

// creating files to save the data
std::ofstream weightDistancesfs("/home/pi/projects/dual_lineFollow/Plotting/weight_distances.csv");
std::ofstream predictor("/home/pi/projects/dual_lineFollow/Plotting/predictor.csv");

bool firstInputs = 1; // used to start the first iteration of learning

// runing the NN on each iteration, each step
double run_samanet(std::vector<double> &predictorDeltas, double error){
  // capturing the time stamp
  using namespace std::chrono;
  milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  std::vector<double> networkInputs;

  predictor << ms.count(); // wrtting the time to the file
  networkInputs.reserve(predictorDeltas.size() * 5); // creating a vector for network inputs

  for (int i =0; i < predictorDeltas.size(); i++){
    predictor << " " << error; // write the error to file
    double sampleValue = predictorDeltas[i];
       
    //cout << sampleValue << ' ';
    
    predictor << " " << sampleValue; // write to file
    predVector1[i].push_back(sampleValue); // add to buffer
    predVector2[i].push_back(sampleValue);
    predVector3[i].push_back(sampleValue);
    predVector4[i].push_back(sampleValue);
    predVector5[i].push_back(sampleValue);

    networkInputs.push_back(predVector1[i][0]); // add to input vector
    networkInputs.push_back(predVector2[i][0]);
    networkInputs.push_back(predVector3[i][0]);
    networkInputs.push_back(predVector4[i][0]);
    networkInputs.push_back(predVector5[i][0]);

    predictor << " " << predVector1[i][0]; // write to file
    predictor << " " << predVector2[i][0];
    predictor << " " << predVector3[i][0];
    predictor << " " << predVector4[i][0];
    predictor << " " << predVector5[i][0];
  }


/*
  for (int j = 0; j < predictorDeltas.size(); ++j) {
    predictor << " " << error;
    double sample = predictorDeltas[j];
    predictor << " " << sample;
    for (auto &filt : bandpassFilters[j]) {
      auto filtered = filt.filter(sample);
      networkInputs.push_back(filtered);
      predictor << " " << filtered;
    }
  } */

  predictor << "\n" ; // new line in file

  // Starting the first propagation, normally the NN has to learn from its previous action then
  // take a new action. Meaning, the backpropagation of feedback takes place first, then the NN
  // generates a new output through forward propagation of the predictors. However, for the very first
  // iteration one forward propagation is needed (first act), then the iterations will be  in pairs of
  // (learn from 1st act, do 2nd act) (learn from 2nd act, do 3rd act) (learn from 3rd, do 4th act) ...
  if (firstInputs == 1){
    samanet->setInputs(networkInputs.data());
    samanet->propInputs();
    firstInputs = 0;
  }
  assert(std::isfinite(error)); // making sure that the error is finite number
  
  // NN has 6 different errors in different pipelines
  // The second one is for the Back-propagated error
  // a coefficient of 1 ensures that the back-propagated error is in effect
  // coefficient of 0 ensure that other ones are not in effect 
  samanet->setErrorCoeff(0,1,0,0,0,0); 

  samanet->setGlobalError(error); // This is the overal error that is sent to all neurons
  samanet->setBackwardError(error); // This is the error that is used for back-propagation
  samanet->propErrorBackward(); // This propagated the error back through the NN
  
  samanet->updateWeights(); // Learn from previous action, this updates the weights
  samanet->setInputs(networkInputs.data()); //then take a new action, set the inputs
  samanet->propInputs(); // propagates the inputs
  samanet->snapWeights();
 
 /* 
  cout << "BCL NEtwork Inputs: " << '\n' << endl;
  for (int i =0; i < networkInputs.size(); i++){
   cout << networkInputs[i] << ' ';
  }
  cout << "\n" << endl;
  */
  
  // saving the weights into the file
  double compensationScale = 1;
  for (int i = 0; i <numLayersBCL; i++){
    if (i == 0){ // for the first layer the weight change is amplified so that it is more visible in plots
      compensationScale = 0.01; // compensates for the weight change amplification
    }
    weightDistancesfs << compensationScale * samanet->getLayerWeightDistance(i) << " ";
    compensationScale = 1;
  }
  
   weightDistancesfs << 0.01 * samanet->getWeightDistance() << "\n";

  double coeff[3] = {1,3,5}; // This are the weights for the 3 outputs of the network
  // 3 different outputs are sumed in a weighted manner
  // so that the NN can output slow, moderate, or fast steering
  double outSmall = samanet->getOutput(0);
  double outMedium = samanet->getOutput(1);
  double outLarge = samanet->getOutput(2);
  double resultNN = (coeff[0] * outSmall) + (coeff[1] * outMedium) + (coeff[2] * outLarge);
  //cout << resultNN << '\n' <<endl;
  return resultNN; // returns the overall output of the NN
  // which together with the reflex error drives the robot's navigation
}


//############################################# FCL Running for each iteration ##############################################

double run_fclNet(std::vector<double> &predictorDeltas, double reflex_error, float* vR, float* vL){
  // capturing the time stamp
  using namespace std::chrono;
  milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  //std::vector<double> networkInputs;
  assert(std::isfinite(reflex_error)); // making sure that the error is finite number 
 
  //initialising input arrays 
  fclInputsArray= new double[predictorDeltas.size()];
	reflex_errorArray = new double[nNeuronsInLayers[0]];
 
  
 for (int i = 0; i< nNeuronsInLayers[0]; i++){ //setting up reflex_error array
  reflex_errorArray[i] = reflex_error;
  //cout << "Reflex Error Inputs: " << reflex_errorArray[i] << '\n' << endl;  
 }
 
 //cout << "preddelts: " << endl; 
 for (int i =0; i < predictorDeltas.size(); i++){ //setting up the network inputs 
   fclInputsArray[i] = predictorDeltas[i]*100;
   //cout << predictorDeltas[i] << ' ';
 }
  
 fclFB->doStep(fclInputsArray,reflex_errorArray);

 //Overwriting the FCL weight distances to the same file used for BCL
  for (int i = 0; i <numLayersFCL; i++){
  
		weightDistancesfs << fclFB->getLayer(i)->getWeightDistanceFromInitialWeights() << " ";
		
  }
  weightDistancesfs << "\n";
  

 //Seperate left and right motor commands from network: 
 //  
  		 *vL = (float)((fclFB->getOutputLayer()->getNeuron(0)->getOutput())*50 +
  				   (fclFB->getOutputLayer()->getNeuron(1)->getOutput())*10 +
  				   (fclFB->getOutputLayer()->getNeuron(2)->getOutput())*2);
  		 *vR = (float)((fclFB->getOutputLayer()->getNeuron(3)->getOutput())*50 +
  				   (fclFB->getOutputLayer()->getNeuron(4)->getOutput())*10 +
  				   (fclFB->getOutputLayer()->getNeuron(5)->getOutput())*2);
  //cout << "vL =: " << vL << endl;

 // Combining motor commands so Left and right have equal commands (matching the BCL algo):

  //double coeff[3] = {1,3,5}; // This are the weights for the 3 outputs of the network
  
  // 3 different outputs are sumed in a weighted manner
  // so that the NN can output slow, moderate, or fast steering
 
 // double outSmall = fclFB->getOutputLayer()->getNeuron(0)->getOutput();
 // double outMedium = fclFB->getOutputLayer()->getNeuron(1)->getOutput();
 // double outLarge = fclFB->getOutputLayer()->getNeuron(2)->getOutput();
  
  double outTest = fclFB->getLayer(1)->getNeuron(3)->getOutput();
   
  //double resultNN = (coeff[0] * outSmall) + (coeff[1] * outMedium) + (coeff[2] * outLarge);
  
  //cout << "nnFclOut: " << resultNN << '\n' <<endl;
  cout << "testNeuronOutput: " << outTest << '\n' <<endl;
  //cout << "Number of inputs : " << fclFB->getNumInputs() << endl;
  
  //return resultNN; // returns the overall output of the NN which together with the reflex error drives the robot's navigation
}

void fcl_weightPlotting(){

  char tmp[256];
  sprintf(tmp,"FCLwL.dat");
  fclFB->getLayer(0)->saveWeightMatrix(tmp);
}
  
