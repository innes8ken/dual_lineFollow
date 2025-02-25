#include "layer.h"
#include "neuron.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <iostream>
#include <string>
#include <initializer_list>
#include <fstream> 


#include <assert.h>
#include <ctgmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <numeric>


using namespace std;

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <paul@glasgowneuro.tech>
 **/

FCLLayer::FCLLayer(int _nNeurons, int _nInputs) {

	nNeurons = _nNeurons;
	nInputs = _nInputs;
	normaliseWeights = WEIGHT_NORM_NONE;

	neurons = new FCLNeuron*[nNeurons];

	calcOutputThread = new CalcOutputThread*[NUM_THREADS];
	learningThread = new LearningThread*[NUM_THREADS];
	maxDetThread = new MaxDetThread*[NUM_THREADS];

	int neuronsPerThread = nNeurons/NUM_THREADS+1;
	for(int i=0;i<NUM_THREADS;i++) {
		calcOutputThread[i] = new CalcOutputThread(neuronsPerThread);
		learningThread[i] = new LearningThread(neuronsPerThread);
		maxDetThread[i] = new MaxDetThread(neuronsPerThread);
	}

	for(int i=0;i<nNeurons;i++) {
		neurons[i] = new FCLNeuron(nInputs);
		calcOutputThread[i%NUM_THREADS]->addNeuron(neurons[i]);
		learningThread[i%NUM_THREADS]->addNeuron(neurons[i]);
		maxDetThread[i%NUM_THREADS]->addNeuron(neurons[i]);
	}

	initWeights(0,0,FCLNeuron::CONST_WEIGHTS);

}

FCLLayer::~FCLLayer() {
	for(int i=0;i<nNeurons;i++) {
		delete neurons[i];
	}
	delete [] neurons;

	for(int i=0;i<NUM_THREADS;i++) {
		delete calcOutputThread[i];
		delete learningThread[i];
		delete maxDetThread[i];
	}

	delete [] calcOutputThread;
	delete [] learningThread;
	delete [] maxDetThread;
	
}

void FCLLayer::calcOutputs() {
	if (useThreads) {
		//fprintf(stderr,"+");
		for(int i=0;i<NUM_THREADS;i++) {
			calcOutputThread[i]->start();
		}
		for(int i=0;i<NUM_THREADS;i++) {
			calcOutputThread[i]->join();
		}
	} else {
		//fprintf(stderr,"-");
		for (int i = 0; i<nNeurons; i++) {
			neurons[i]->calcOutput();
		}
	}
}

void FCLLayer::doNormaliseWeights() {
	double norm = 0;
	switch (normaliseWeights) {
	case WEIGHT_NORM_LAYER_EUCLEDIAN:
		for(int i=0;i<nNeurons;i++) {
			norm = norm + neurons[i]->getSumOfSquaredWeightVector();
		}
		norm = sqrt(norm);
		for(int i=0;i<nNeurons;i++) {
			neurons[i]->normaliseWeights(norm);
		}
		break;
	case WEIGHT_NORM_NEURON_EUCLEDIAN:
		for(int i=0;i<nNeurons;i++) {
			norm = neurons[i]->getEuclideanNormOfWeightVector();
			neurons[i]->normaliseWeights(norm);
		}
		break;
	case WEIGHT_NORM_LAYER_MANHATTAN:
		for(int i=0;i<nNeurons;i++) {
			norm = norm + neurons[i]->getManhattanNormOfWeightVector();
		}
		for(int i=0;i<nNeurons;i++) {
			neurons[i]->normaliseWeights(norm);
		}
		break;
	case WEIGHT_NORM_NEURON_MANHATTAN:
		for(int i=0;i<nNeurons;i++) {
			norm = neurons[i]->getManhattanNormOfWeightVector();
			neurons[i]->normaliseWeights(norm);
		}
		break;
	case WEIGHT_NORM_LAYER_INFINITY:
		for(int i=0;i<nNeurons;i++) {
			double a = neurons[i]->getInfinityNormOfWeightVector();
			if (a>norm) norm = a;
		}
		for(int i=0;i<nNeurons;i++) {
			neurons[i]->normaliseWeights(norm);
		}
		break;
	case WEIGHT_NORM_NEURON_INFINITY:
		for(int i=0;i<nNeurons;i++) {
			norm = neurons[i]->getInfinityNormOfWeightVector();
			neurons[i]->normaliseWeights(norm);
		}
		break;
	default:
		break;
	}
}

void FCLLayer::doLearning() {
	if (useThreads) {
		if (maxDetLayer) {
			for(int i=0;i<NUM_THREADS;i++) {
				maxDetThread[i]->start();
			}
		} else {
			//fprintf(stderr,"*");
			for(int i=0;i<NUM_THREADS;i++) {
				learningThread[i]->start();
			}
		}
		if (maxDetLayer) {
			for(int i=0;i<NUM_THREADS;i++) {
				maxDetThread[i]->join();
			}
		} else {
			for(int i=0;i<NUM_THREADS;i++) {
				learningThread[i]->join();
			}
		}
	} else {
		if (maxDetLayer) {
			for (int i = 0; i<nNeurons; i++) {
				neurons[i]->doMaxDet();
			}
		}
		else {
			fprintf(stderr,"_");
			for (int i = 0; i<nNeurons; i++) {
				neurons[i]->doLearning();
			}
		}
	}
	doNormaliseWeights();
}

void FCLLayer::setNormaliseWeights(WeightNormalisation _normaliseWeights) {
	normaliseWeights = _normaliseWeights;
	for(int i=0;i<nNeurons;i++) {
		doNormaliseWeights();
		neurons[i]->saveInitialWeights();
	}	
}


void FCLLayer::setError(double _error) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setError(_error);
	}
}

void FCLLayer::setErrors( double* _errors) {
	for(int i=0;i<nNeurons;i++) {
		if (isnan(_errors[i])) {
			fprintf(stderr,"Layer::%s L=%d, errors[%d]=%f\n",__func__,layerIndex,i,_errors[i]);
		}
		neurons[i]->setError(_errors[i]);
	}
}

void FCLLayer::setBias(double  _bias) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setBias(_bias);
	}
}

void FCLLayer::setLearningRate( double _learningRate) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setLearningRate(_learningRate);
	}
}

void FCLLayer::setMomentum( double _momentum) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setMomentum(_momentum);
	}
}

void FCLLayer::setActivationFunction(FCLNeuron::ActivationFunction _activationFunction) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setActivationFunction(_activationFunction);
	}
}

void FCLLayer::setDecay( double _decay) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setDecay(_decay);
	}
}

void FCLLayer::initWeights( double max, int initBias, FCLNeuron::WeightInitMethod weightInitMethod) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->initWeights(max,initBias,weightInitMethod);
	}
}

void FCLLayer::setError( int i,  double _error) {
	assert(i < nNeurons);
	neurons[i]->setError(_error);
}

double FCLLayer::getError( int i) {
	assert(i < nNeurons);
	return neurons[i]->getError();
}

// setting a single input to all neurons
void FCLLayer::setInput(int inputIndex, double input) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setInput(inputIndex,input);
	}
}

// setting a single input to all neurons
void FCLLayer::setDebugInfo(int _layerIndex) {
	layerIndex = _layerIndex;
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setDebugInfo(_layerIndex,i);
	}
}


// setting a single input to all neurons
void FCLLayer::setStep(long int _step) {
	step = _step;
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setStep(step);
	}
}

double FCLLayer::getWeightDistanceFromInitialWeights() {
	double distance = 0;
	for(int i=0;i<nNeurons;i++) {
		distance += neurons[i]->getWeightDistanceFromInitialWeights();
	}
	return distance;
}


void FCLLayer::setInputs( double* inputs ) {
	double* inputp = inputs;
	inputp = inputs;
	for(int j=0;j<nInputs;j++) {
		FCLNeuron** neuronsp = neurons;
		 double input = *inputp;
		inputp++;
		for(int i=0;i<nNeurons;i++) {
			(*neuronsp)->setInput(j,input);
			neuronsp++;
		}
	}

}


void FCLLayer::setConvolution( int width,  int height) {
	double d = round(sqrt(nNeurons));
	int dx = (int)round(width/d);
	int dy = (int)round(height/d);
	int mx = (int)round(dx/2.0);
	int my = (int)round(dy/2.0);
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setGeometry(width,height);
		neurons[i]->setMask(0);
		for(int x=0;x<dx;x++) {
			for(int y=0;y<dy;y++) {
				neurons[i]->setMask(x+mx-dx/2,y+my-dx/2,1);
			}
		}
		mx = mx + dx;
		if (mx > width) {
			mx = (int)round(dx/2.0);
			my = my + dy;
		}
	}
}


int FCLLayer::saveWeightMatrix() {
	std::ofstream wfileFCL("/home/pi/projects/dual_lineFollow/Plotting/FCLwL.csv");
	
	//if (!filename) return errno;
	for(int i=0;i<nNeurons;i++) {
		for(int j=0;j<neurons[i]->getNinputs();j++) {
			wfileFCL << neurons[i]->getWeight(j) << ' ';
		}
		wfileFCL << "\n";
	}
	//wfileFCL.close();
	return 0;
	
	//FILE* f = fopen(filename,"FCLwL"); //wt before wL1
	//if (!f) return errno;
	//for(int i=0;i<nNeurons;i++) {
		//for(int j=0;j<neurons[i]->getNinputs();j++) {
			//fprintf(f,"%f\t",neurons[i]->getWeight(j));
		//}
		//fprintf(f,"\n");
	//}
	//fclose(f);
	//return 0;
}
