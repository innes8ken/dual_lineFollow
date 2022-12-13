#include "opencv2/opencv.hpp"
#include <boost/circular_buffer.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <numeric>
#include "neural.h"
#include "external.h"
#include "cvui.h"
#include "LowPassFilter.hpp"
#include "bandpass.h"
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>
//#include "FCL/fcl_util.h"
//#include "FCL/fcl.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

Extern::Extern(){

}
int samplingFreq = 30; // 30Hz is the sampling frequency
int figureLength = 5; //seconds

//create buffers for plotting
boost::circular_buffer<double> reflex_error_plot(samplingFreq * figureLength);
boost::circular_buffer<double> sensor0(samplingFreq * figureLength);
boost::circular_buffer<double> sensor1(samplingFreq * figureLength);
boost::circular_buffer<double> sensor2(samplingFreq * figureLength);
boost::circular_buffer<double> sensor3(samplingFreq * figureLength);
boost::circular_buffer<double> sensor4(samplingFreq * figureLength);
boost::circular_buffer<double> sensor5(samplingFreq * figureLength);
boost::circular_buffer<double> sensor6(samplingFreq * figureLength);
boost::circular_buffer<double> sensor7(samplingFreq * figureLength);
boost::circular_buffer<double> reflex_error_moving_ave_plot(samplingFreq * 100 * figureLength);
// creating files to save the data
std::ofstream datafs("/home/pi/projects/dual_lineFollow/Plotting/speedDiffdata.csv");
std::ofstream modulusFile("/home/pi/projects/dual_lineFollow/Plotting/modulusData.csv");


//##################################   Mutual Environment   ######################################################################
double reflex_error_gain = 1.9; // reflex error's gain, how much influence the reflex has on the steering 
double nn_output;
double nn_left_output;
double nn_right_output;
double nn_gain_coeff; 
double nn_gain_power;

//##################################   BCL Environment   #########################################################################
// NN gain is calculated as coeff x 10^(power)
double bcl_nn_gain_coeff = 1.2; // NN'output gain for steering, the coefficient
double bcl_nn_gain_power = 0; // NN'output gain for steering, the power of 10

//###################################    FCL Environment  ########################################################################
double fcl_nn_gain_coeff = 1.2; // NN's output parameters for steering: coefficient * 10^(power)
double fcl_nn_gain_power = 0;

//################################################################################################################################

// The 'onStepCompleted' is called every step 

int Extern::onStepCompleted(cv::Mat &stat_frame, double reflex_error, std::vector<double> &predictorDeltaMeans_, int paradigmOption_, double *leftCommand, double *rightCommand) {
  assert(std::isfinite(reflex_error)); // making sure that the reflex error is finite value
  reflex_error_plot.push_back(reflex_error); //puts the reflex errors in a buffer for plotting
  
  // # SECTION: NN LEARNING
  /**
   * Making a copy of the reflex error.
   * 'reflex_error' is used for motor command
   * 'feedback_error' is used for training the network in the BCL algo
   * This allows us to work with the reflex and the learning separately
   **/
  double feedback_error = reflex_error;
  /**
   * If the learning gain is zero, the feedback to the NN is set to zero too
   * So no learning takes place, meaning, no changes to the weights.
   **/
  
  if ((bcl_nn_gain_coeff == 0)||(fcl_nn_gain_coeff == 0)){
    feedback_error = 0;
  }

   /**
   * Switch case to Run both Sama's net and FCL algo from neural.cpp. Does one iteration of learning
   * Pass in the predictors differences (before filtration)
   * Pass in the feedback_error (same as reflex error unless the learning is off)
   * Returns the NN's output, which is a weighted sum of neurons' output in the last layer
   **/
   
  switch (paradigmOption_){
  case 0:
  nn_output = run_samanet(predictorDeltaMeans_, feedback_error); // the output of one iteration through BCL learning 
  nn_gain_coeff = bcl_nn_gain_coeff;  // Setting the nn gain varirables 
  nn_gain_power = bcl_nn_gain_power;
  
  break;

  case 1:
  //cout<< "Size of FCL predictor input array: " << predictorDeltaMeans_.size()<< endl;
  //cout<< "Size of FCL reflex input array: " << sizeof(predictorDeltaMeans_)/sizeof(*predictorDeltaMeans_) << endl;
  nn_output = run_fclNet(predictorDeltaMeans_, reflex_error, &nn_left_output, &nn_right_output); // the output of one iteration of FCL learning 
  nn_gain_coeff = fcl_nn_gain_coeff;
  nn_gain_power = fcl_nn_gain_power;
  
  break;
  }

  // ###########################################   GENERAL MOTOR COMMANDS   ##############################################################
  double reflex_for_nav = reflex_error * reflex_error_gain; // calculate the relfex part of speed command
  double learning_for_nav = nn_output * nn_gain_coeff * pow(10,nn_gain_power); // calculate the learning part of the motor speed command
 // cout<<"Learning for Nav: " << learning_for_nav<<endl;
  cout<< "NN Comand result: "<<nn_output<<endl;
  
  double motor_command = reflex_for_nav + learning_for_nav; // calculate the overal motor command used for BCL robot control 
  *leftCommand = (nn_right_output * nn_gain_coeff * pow(10,nn_gain_power) + reflex_for_nav); // Seperate wheel commands to be used for FCL robot control
  *rightCommand = (nn_left_output * nn_gain_coeff * pow(10,nn_gain_power) + reflex_for_nav);


  // ###########################################   SECTION: PLOTS   ######################################################################
  /**
   * setting up the GUI with the stats. Display the stat
   * Setting up track bars for the reflex and learning gains to be changes in trials interactively
   **/
  std::vector<double> error_list(reflex_error_plot.begin(), reflex_error_plot.end());
  cvui::text(stat_frame, 10, 250, "Sensor Error Multiplier: ");
  cvui::trackbar(stat_frame, 180, 250, 400, &reflex_error_gain, (double)0.0, (double)10.0, 1, "%.2Lf", 0, 0.5);
  cvui::text(stat_frame, 10, 300, "Net Output Multiplier: ");
  cvui::trackbar(stat_frame, 180, 300, 400, &nn_gain_coeff, (double)0.0, (double)10.0, 1, "%.2Lf", 0, 0.5);
	cvui::trackbar(stat_frame, 180, 350, 400, &nn_gain_power, (double)0, (double)20, 1, "%.2Lf", 0, 0.5);
  cvui::sparkline(stat_frame, error_list, 10, 50, 580, 200, 0x000000); //Black = reflex_error
  cvui::text(stat_frame, 220, 10, "Net out:");
  cvui::printf(stat_frame, 300, 10, "%+.4lf (%+.4lf)", nn_output, learning_for_nav);
  cvui::text(stat_frame, 220, 30, "Error:");
  cvui::printf(stat_frame, 300, 30, "%+.4lf (%+.4lf)", reflex_error, reflex_for_nav);

  // saving the data into files
  datafs << reflex_error << " "
         << reflex_error << " "
         << reflex_for_nav << " "
         << nn_output << " "
         << learning_for_nav << " "
         << motor_command << "\n";
  
  // return the motor command as an int type to be written to arduino
  return (int)motor_command;
}

// create filters for sensor datann_output
double cutOff = 10;
double sampFreq = 0.033;
Bandpass sensorFilters[8];
LowPassFilter lpf0(cutOff, sampFreq);
LowPassFilter lpf1(cutOff, sampFreq);
LowPassFilter lpf2(cutOff, sampFreq);
LowPassFilter lpf3(cutOff, sampFreq);
LowPassFilter lpf4(cutOff, sampFreq);
LowPassFilter lpf5(cutOff, sampFreq);
LowPassFilter lpf6(cutOff, sampFreq);
LowPassFilter lpf7(cutOff, sampFreq);

// each loop of path is about 1500 samples
// create an arrray to monitor the moving average of the error
const int loopLength = 400;
boost::circular_buffer<double> movingIntegralVector(loopLength);

// create flags for monitoring the trial and the learning condition
int checkSucess = 0;
int consistency = 0;
int stepCount = 1;
int successDone = 0;

// creating files for learning
std::ofstream errorSuccessDatafs("/home/pi/projects/dual_lineFollow/Plotting/errorSuccessData.csv");
std::ofstream successRatef("/home/pi/projects/dual_lineFollow/Plotting/successTime.csv");

// initialise some variables
int sensorInUse = 6;
double thresholdInteg = 0;
int getThreshold = 1;
double maxMovingIntegral = 0;
double maxforThreshold = 0.00;
int setFirstEncounter = 1;
int firstEncounter = 0;
double totalIntegral = 0;

/**
 * 'calcError' is called from the main.cpp file.
 * It takes in the state_frame where the info will be displayed 
 * It also takes in the raw sensor data in order to calculate the reflex error
 **/
double Extern::calcError(cv::Mat &stat_frame, vector<uint8_t> &sensorCHAR, int paradigmOption_){
	const int numSensors = 8; //number of sensors
	int startIndex = 8; // used to find the first sensor data, syncing the data
	int sensorINT[numSensors+1]= {0,0,0,0,0,0,0,0,0}; // empty array to read the data in
  /**
   * There are 8 sensors, but 9 numbers are sent, the zero marks the start of the serial communication
   * If the value of an index is zero, then the next index is the start index of sensor data 
   **/
	for (int i = 0; i < numSensors+1 ; i++){
		sensorINT[i] = (int)sensorCHAR[i];
		if (sensorINT[i] == 0){
			startIndex = i + 1;
		}
	}
  // create a new array for sensor data with type double
  double sensorVAL[numSensors+1]= {0,0,0,0,0,0,0,0,0};
  // # SECTION: MAP SENSOR DATA to [50,250]
  double mapBlack = 50; //y1
  double mapWhite = 250; //y2
  double m [8+1] = {1,1,1,1,1,1,1,1,1};
  //char colorName[8] = {'R', 'O', 'Y', 'G', 'B', 'V', 'P', 'W'}; // Red,Orange,Yellow,Green,Dark-Green,Purple,L-Blue,D-Blue
  //const int colorCode[8] = {0xff0000, 0xff9900, 0xffff00, 0x00ff00, 0x00ffff, 0x9900ff, 0xff00ff, 0xffffff};

  /**
   * Mapping the sensor data to the 50-250 range.
   * If the sensors do not read the white/black background as expected
   * change the thresholds in external.h files to adjust to the new lighting/environment
   **/
  for (int i = 0; i < numSensors; i++){
    int remainIndex = (startIndex + i) % (numSensors+1);
    sensorVAL[i] = sensorINT[remainIndex];
    if (sensorVAL[i] > threshWhite[i] ){calibWhite[i] = sensorVAL[i];}
    if (sensorVAL[i] < threshBlack[i] ){calibBlack[i] = sensorVAL[i];}
    diffCalib[i] = calibWhite[i] - calibBlack[i];
    assert(std::isfinite(diffCalib[i]));
    m[i] = (mapWhite - mapBlack)/(diffCalib[i]);
    sensorVAL[i] = m[i] * (sensorINT[remainIndex] - calibBlack[i]) + mapBlack;
    assert(std::isfinite(sensorVAL[i]));
    // cout << colorName[i] << " Bcal: " << (int)calibBlack[i] << " " << (int)threshBlack[i]
    //       << " raw: " << (int)sensorINT[remainIndex]
    //       << " Wcal: " << (int)threshWhite[i] << " " << (int)calibWhite[i]
    //       << " cal: " << (int)sensorVAL[i] << endl;
  }
  //cout << " ------------------------------- "<< endl;

  // filter the mapped sensor data to remove noise
  sensorVAL[0] = lpf0.update(sensorVAL[0]);
  sensorVAL[1] = lpf1.update(sensorVAL[1]);
  sensorVAL[2] = lpf2.update(sensorVAL[2]);
  sensorVAL[3] = lpf3.update(sensorVAL[3]);
  sensorVAL[4] = lpf4.update(sensorVAL[4]);
  sensorVAL[5] = lpf5.update(sensorVAL[5]);
  sensorVAL[6] = lpf6.update(sensorVAL[6]);
  sensorVAL[7] = lpf7.update(sensorVAL[7]);
  
  // weighting the sensor pairs to indicate the degree of deviation
  double sesnor_diff_weights[numSensors/2] = {7,5,3,1};
  double reflex_error = 0; // create a variable for the overal reflex error
  /**
   * Calculate the error as the weighted differences of sensor pairs
   * if you want to exlude some sensors change the for-statement accordingly
   **/
  for (int i = 0 ; i < 2 ; i++){          //change 1< 3 to tailor number of active sensor pairs
      reflex_error += (sesnor_diff_weights[i]) * (sensorVAL[i] - sensorVAL[numSensors -1 -i]);
  }
  reflex_error = reflex_error / (mapWhite - mapBlack); // normalise the error to the range
  assert(std::isfinite(reflex_error)); // make sure the error is a finite value

  //plot the sensor values:
  double minVal = 40; // used to adjust the plots
  double maxVal = 260; // used to sdjust the plots

  // Change sensor colours here 
  sensor0.push_back(sensorVAL[0]); //puts the errors in a buffer for plotting
  sensor0[0] = minVal;
  sensor0[1] = maxVal;
  std::vector<double> sensor_list0(sensor0.begin(), sensor0.end());
  cvui::sparkline(stat_frame, sensor_list0, 10, 50, 580, 200, 0xff0000); // RED

  sensor1.push_back(sensorVAL[1]); //puts the errors in a buffer for plotting
  //cout << "orange sensor: " << sensorVAL[1] << endl;
  sensor1[0] = minVal;
  sensor1[1] = maxVal;
  std::vector<double> sensor_list1(sensor1.begin(), sensor1.end());
  cvui::sparkline(stat_frame, sensor_list1, 10, 50, 580, 200, 0xff7000); // ORANGE

  sensor2.push_back(sensorVAL[2]); //puts the errors in a buffer for plotting
  sensor2[0] = minVal;
  sensor2[1] = maxVal;
  std::vector<double> sensor_list2(sensor2.begin(), sensor2.end());
  cvui::sparkline(stat_frame, sensor_list2, 10, 50, 580, 200, 0xffec00); // YELLOW

  sensor3.push_back(sensorVAL[3]); //puts the errors in a buffer for plotting
  sensor3[0] = minVal;
  sensor3[1] = maxVal;
  std::vector<double> sensor_list3(sensor3.begin(), sensor3.end());
  cvui::sparkline(stat_frame, sensor_list3, 10, 50, 580, 200, 0x32ff00); // GREEN

  sensor4.push_back(sensorVAL[4]); //puts the errors in a buffer for plotting
  sensor4[0] = minVal;
  sensor4[1] = maxVal;
  std::vector<double> sensor_list4(sensor4.begin(), sensor4.end());
  cvui::sparkline(stat_frame, sensor_list4, 10, 50, 580, 200, 0x008202); // DARK GREEN

  sensor5.push_back(sensorVAL[5]); //puts the errors in a buffer for plotting
  sensor5[0] = minVal;
  sensor5[1] = maxVal;
  std::vector<double> sensor_list5(sensor5.begin(), sensor5.end());
  cvui::sparkline(stat_frame, sensor_list5, 10, 50, 580, 200, 0xa600ff); //PURPLE

  sensor6.push_back(sensorVAL[6]); //puts the errors in a buffer for plotting
  sensor6[0] = minVal;
  sensor6[1] = maxVal;
  std::vector<double> sensor_list6(sensor6.begin(), sensor6.end());
  cvui::sparkline(stat_frame, sensor_list6, 10, 50, 580, 200, 0x00ffbd); // LIGHT BLUE 

  sensor7.push_back(sensorVAL[7]); //puts the errors in a buffer for plotting
  sensor7[0] = minVal;
  sensor7[1] = maxVal;
  std::vector<double> sensor_list7(sensor7.begin(), sensor7.end());
  cvui::sparkline(stat_frame, sensor_list7, 10, 50, 580, 200, 0x0027ff); // DARK BLUE

  // average the reflex_error over the last N samples:
  stepCount += 1; // count how many steps the program has taken
  checkSucess += 1; // used to start checking for successful learning after a certain number of steps.
                    // to give time for the reflex error to accumulate
  /**
   * It detects when the robot has encountered the line for the first time
   * This is important to keep the data consistent, so that the idle runing of robot
   * at the start of the trial is not taken into consideration, as this period depends on
   * how quickly the robot has been turned on and the exact location of it on the path.
   * This way, the statistics and the learning starts when the robot detects the line for the first time
   **/
  if (fabs(reflex_error) > 0.01 && setFirstEncounter == 1){
    firstEncounter = stepCount; // this is used to subtract the initial steps before encountering the line
    setFirstEncounter =0; // stops looking for first counter
  }
  movingIntegralVector.push_back(abs(reflex_error)); // buffer for moving integral of error
  // sum the moving integral buffer to get the overal integral
  double movingIntegralSum = std::accumulate(movingIntegralVector.begin(), movingIntegralVector.end(), 0.00);
  double movingIntegralAve = movingIntegralSum/loopLength; // average of integral over the length of buffer
  totalIntegral += abs(reflex_error); // total integral of error over the entire trial
  double totalIntegralAve = totalIntegral/stepCount; // average of error over entire trial
  maxMovingIntegral = max (maxMovingIntegral,fabs(movingIntegralAve)); // max of the two mentioned above
  // setting a threshold for success condition
  // the moving average has to fall below this threshold
  // so that the learning is considered a success
  thresholdInteg = maxMovingIntegral * 0.01;
  
  /**
   * Display the moving average on the GUI
   **/
  reflex_error_moving_ave_plot.push_back(movingIntegralAve); //just for displaying the moving average
  std::vector<double> movingAveErrorList(reflex_error_moving_ave_plot.begin(), reflex_error_moving_ave_plot.end());
  cvui::sparkline(stat_frame, movingAveErrorList, 10, 50, 580, 200, 0xffffff); //white = moving average 

  // save some variable (updating plotting file?)
  errorSuccessDatafs << reflex_error << " "
          << movingIntegralAve << "\n";
  
  int actualSteps = stepCount-firstEncounter; // take away the idle steps before ancountering the line
  
  // display more stats
  cvui::text(stat_frame, 10, 20, "Step:");
  cvui::printf(stat_frame, 40, 20, "%d", actualSteps);
  cvui::text(stat_frame, 10, 10, "Max:");
  cvui::printf(stat_frame, 40, 10, "%1.3f", maxMovingIntegral);
  cvui::text(stat_frame, 100, 10, "Thsh:");
  cvui::printf(stat_frame, 130, 10, "%1.3f", thresholdInteg);
  cvui::text(stat_frame, 100, 20, "ave:");
  cvui::printf(stat_frame, 130, 20, "%1.3f", movingIntegralAve);

  if (nn_gain_coeff == 0){ 
    // this is for reflex
    // stop the program if certain number of steps has been taken
    if ( stepCount - firstEncounter > loopLength * 2 && successDone == 0){
        cout << "DONE!" << endl;
        cout << "movingIntegralAve: " << movingIntegralAve << endl;
        cout << "maxMovingIntegral: " << maxMovingIntegral << endl;
        cout << "totalIntegral: " << totalIntegral << endl;
        cout << "totalIntegralAve: " << totalIntegralAve << endl;
      successDone = 1;
      successRatef << firstEncounter << " " << stepCount - firstEncounter 
                  << " " << movingIntegralAve << " " << maxMovingIntegral 
                  << " " << totalIntegral << " " << totalIntegralAve << "\n";
      exit(19);
    }
  }else{ // this is for learning
    // sets the error to 100 if the success condition has been met
    // This then stops the infinite loop in the main.cpp file and stops the program
    if (checkSucess > firstEncounter + loopLength/2 && fabs(movingIntegralAve) < thresholdInteg && successDone == 0){
      //start checking for success 100 steps after it has seen the line first
      consistency += 1;
      if (consistency > 10){ // the success condition has to persist for 10 steps
        cout << "SUCCESS! on Step: " << stepCount - firstEncounter << endl;
        //cout << "movingIntegralAve: " << movingIntegralAve << endl;
        //cout << "maxMovingIntegral: " << maxMovingIntegral << endl;
        //cout << "totalIntegral: " << totalIntegral << endl;
        //cout << "totalIntegralAve: " << totalIntegralAve << endl;
         
        
        if (paradigmOption_ == 1){fcl_weightPlotting();} //This is to record and later plot the weight values of the nn in the first layer
      
        successDone = 1;
        successRatef << firstEncounter << " " << stepCount - firstEncounter 
                  << " " << movingIntegralAve << " " << maxMovingIntegral
                  << " " << totalIntegral << " " << totalIntegralAve << "\n";
        reflex_error = 100;                            
      }
      
      //exit(14);
        
    }else{consistency = 0;}
  }
  return reflex_error; // return the relfex error for NN feedback and for motor command
}

// set up variables for the predictiors
// number of rows and colomns for the camera view
// number of pixle clusters gives the number of predictors
static constexpr int nPredictorCols = 6;
static constexpr int nPredictorRows = 8;
static constexpr int nPredictors = nPredictorCols * nPredictorRows;

// returns the number of predictors to the main.cpp file
int Extern::getNpredictors (){
    return nPredictors;
}

// 'calcPredictors' calculate the predictors for the NN  
// Should work for both paradigms 
void Extern::calcPredictors(Mat &frame, vector<double> &predictorDeltaMeans){
	// Define the rectangular area that will be separated from the full camera view.
  // Only this area will be used for the learning
    int areaWidth = 600;
    int areaHeight = 120;
    int offsetFromTop = 350;
    // VERTICAL RESOLUTION OF CAMERA SHOULD ADJUST

    // sectioning the camera view into smaller areas, the pixle clusters
    int startX = (640 - areaWidth) / 2;
    auto area = Rect{startX, offsetFromTop, areaWidth, areaHeight};
    int predictorWidth = area.width / 2 / nPredictorCols;
    int predictorHeight = area.height / nPredictorRows;
	    Mat edges;
	    cvtColor(frame, edges, COLOR_BGR2GRAY);
  rectangle(edges, area, Scalar(122, 144, 255));
  predictorDeltaMeans.clear();
	int areaMiddleLine = area.width / 2 + area.x;
  // Define the threshold for each pixle cluster, any value greater than this will be considered 
  // as the black path
  double predThreshW[nPredictorCols][nPredictorRows] = {{170,180,190,200,210,220,220,220},
                                                        {170,180,190,190,200,210,220,210},
                                                        {165,170,180,190,190,200,200,200},
                                                        {155,160,170,180,180,180,190,190},
                                                        {145,150,150,160,160,170,170,170},
                                                        {140,140,140,140,140,140,140,140}};
  double predThreshWAdjustment = 20;
  double predThreshWDiff = 50;
  double activePixel = 0.00;
  int index = 0;
	for (int k = 0; k < nPredictorRows; ++k) {
      for (int j = 0; j < nPredictorCols ; ++j) {
         auto lPred =
            Rect(areaMiddleLine - (j + 1) * predictorWidth,
                 area.y + k * predictorHeight, predictorWidth, predictorHeight);
        auto rPred =
            Rect(areaMiddleLine + (j)*predictorWidth,
                 area.y + k * predictorHeight, predictorWidth, predictorHeight);
        auto grayMeanL = mean(Mat(edges, lPred))[0];
        auto grayMeanR = mean(Mat(edges, rPred))[0];
        if (grayMeanL < predThreshW[j][k] - predThreshWDiff){grayMeanL = predThreshW[j][k] - predThreshWDiff;}
        if (grayMeanR < predThreshW[j][k] - predThreshWDiff){grayMeanR = predThreshW[j][k] - predThreshWDiff;}
        if (grayMeanL > predThreshW[j][k] - predThreshWAdjustment){grayMeanL = predThreshW[j][k] - predThreshWAdjustment;}
        if (grayMeanR > predThreshW[j][k] - predThreshWAdjustment){grayMeanR = predThreshW[j][k] - predThreshWAdjustment;}
        double predScale = 0.05;
        auto predValue = ((grayMeanL - grayMeanR) / predThreshWDiff) * predScale;
        if(k == 0 && abs(predValue) > activePixel){
            activePixel = predValue;
            index = j;
        }
        // predictor values are calculated above and saved into 'predictorDeltaMeans'
        predictorDeltaMeans.push_back(predValue);
        // showing the stat in the camera view frame
        putText(frame, std::to_string((int)(grayMeanL - grayMeanR)),
                Point{lPred.x + lPred.width / 2 - 13,
                      lPred.y + lPred.height / 2 + 5},
                FONT_HERSHEY_TRIPLEX, 0.4, {0, 0, 0});
        putText(frame, std::to_string((int)grayMeanR),
                Point{rPred.x + rPred.width / 2 - 13,
                      rPred.y + rPred.height / 2 + 5},
                FONT_HERSHEY_TRIPLEX, 0.4, {0, 0, 0});
        rectangle(frame, lPred, Scalar(50, 50, 50));
        rectangle(frame, rPred, Scalar(50, 50, 50));
      }
    }

    line(frame, {areaMiddleLine, 0}, {areaMiddleLine, frame.rows}, Scalar(50, 50, 255));
    imshow("robot view", frame); // refresh and show the frame
}
