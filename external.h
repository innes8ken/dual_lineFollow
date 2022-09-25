#pragma once
#include <vector>

#include "opencv2/opencv.hpp"

#include <boost/circular_buffer.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "cvui.h"

using namespace cv;
using namespace std;

/**
 * @brief Contains functions which manage error signals, reflex commands and GUI
 * 
 */

class Extern {
	public:

	Extern();

	/**
	 * @brief Returns the overall motor command
	 * 
	 * onStepCompleted is called every timestep where it will complete one step of
	 * learning and sum with reflex signals.  
	 * 
	 * @param stat_frame Used for plotting ?
	 * @param deltaSensorData this is the reflex error found from calling calcError()
	 * @param predictorDeltas the predictors differences
	 * @return int motor_command
	 */
	int onStepCompleted(Mat &stat_frame, double deltaSensorData, vector<double> &predictorDeltas);

	/**
	 * @brief Returns reflex error 
	 * 
	 * Contains mapping, filtering and GUI plotting actions for the reflex signals.
	 * 
	 * @param statFrame Used for plotting?
	 * @param sensorCHAR holds the raw sensor data for the PR sensors 
	 * @return double reflex_Error
	 */
	double calcError(Mat &statFrame, vector<uint8_t> &sensorCHAR);

	/**
	 * @brief Returns the number of predictors to main.cpp
	 * 
	 * handles the predictive signals from the camera array 
	 * 
	 * @param frame 
	 * @param predictorDeltaMeans camera array values 
	 */
	void calcPredictors(Mat &frame, vector<double> &predictorDeltaMeans);

	/**
	 * @brief Get the Npredictors object
	 * 
	 * Returns the number of predictors
	 * 
	 * @return int nPredictors
	 */
	int getNpredictors();

	private:
	using clk = std::chrono::system_clock;
	clk::time_point start_time;

	double calibBlack[8+1]  = {100,110,115,125,   125,120,110,100,0};//x1 Red,Orange,Yellow,Green,Blue,Violet,Pink,White
	double threshBlack[8+1] = {135,140,140,150,   150,140,140,135,0};
	double threshWhite[8+1] = {140,145,145,155,   155,145,145,140,1};
	double calibWhite[8+1]  = {150,160,160,160,   160,160,160,150,2}; //x2 Red,Orange,Yellow,Green,Blue,Violet,Pink,White
	double diffCalib[8+1]   = {1,1,1,1,           1,1,1,1,1};
};
