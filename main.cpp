#include "opencv2/opencv.hpp"
#include "serialib.h"
#include <boost/circular_buffer.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include "neural.h"
#include "external.h"
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#if defined(_WIN32) || defined(_WIN64)
#define DEVICE_PORT "COM4" // COM1 for windows
#endif

#ifdef __linux__
#define DEVICE_PORT "/dev/ttyUSB0" // This is for Arduino, ttyS0 for linux, otherwise ttyUSB0
#endif

#define STAT_WINDOW "statistics & options"

using namespace cv;
using namespace std;

constexpr int ESC_key = 27;
const int numSens9 = 9;

/**
 * Main function of robot follower 
 **/

int main(int n, char* args[]) {

  /**
   * Introducing command line argument paradigmOption
   **/
  string paradigmOption = args[1];

  /**
   * Checking for input error
   **/
  if ((n != 2) || ((paradigmOption!="F") && (paradigmOption!="B")))  
    { 
    cout << "ERROR: Wrong arguments as input. PLease input 'F' OR 'B' after executable" << endl;  
    cout << "Number of input arguments = :" << n -1 << endl;  
    break;
    }
  /**
   * Printing successful paradigm setting  
   **/
  if (paradigmOption == "F"){
    cout << "The learning paradigm is set to Forward error prop"<< endl;
  
  } else if (paradigmOption == "B") {
    cout << "The learning paradigm is set to Backward error prop"<< endl;
  }


  /**
   * Make an (a pointer) instance of the class Extern 
   **/
  Extern* external = NULL;
  external = new Extern();
  /**
   * It returns the number of Predictors
   * This is the number of pixle clusters in the camera view
   * It is calculated by multiplying the number of rows and columns
   **/
  int nPredictors = external->getNpredictors();
  srand(4); //random number generator
  /**
   * Setting up the GUI window
   **/
  cv::namedWindow("robot view");
  cvui::init(STAT_WINDOW);
  auto statFrame = cv::Mat(400, 600, CV_8UC3);
  /**
   * Initialises the CLDL network
   * This is called from the nueral.cpp file
   **/

 switch (ParadigmOption) { //***************************************************************************ADDITION***********************************
   case "B":
    initialize_samanet(nPredictors);
    break 
   //case 1: Initialise FCL 

  /**
   * Setting up a serial communication for arduino
   **/
  serialib LS;
  char Ret = LS.Open(DEVICE_PORT, 115200);
  /**
   * Capture the return from the seriallib 'LS' to detect errors with the serial communication
   * If Ret is not 1, there is an error
   * It returns the value 'Ret' to the main program to exit the application
   **/
  if (Ret != 1) { // If an error occured...
  printf("Error while opening port. Permission problem?\n try: sudo chmod 666 /dev/ttyS0 or ttyUSB0\n");
  return Ret; // ... quit the application
  }
  /**
   * Start the communication without sending the actual data
   **/
  char startChar = {'d'};
  Ret = LS.Write(&startChar, sizeof(startChar));
  printf("Serial port opened successfully !\n");
  /**
   * Start the camera and capture the view
   * 0 for Rpi camera, for other cameras use the correct digit
   **/
  VideoCapture cap(0);
  if (!cap.isOpened()) {
  printf("The selected video capture device is not available.\n");
  return -1;
  }
  /**
   * Create a vector to put the predictor values in
   * It is type 'double'
   * These the GSV values of the pixle clusters in the camera view
   * They are received form the camera and are sent to the CLDL NN for prediction
   **/
  std::vector<double> predictorDeltaMeans;
  predictorDeltaMeans.reserve(nPredictors);  
  /**
   * Create vectors to put the sensor values in
   * This is type unsigned integer 8
   * These are the reflex sensory inputs that are received form the arduino 
   **/
  std::vector<uint8_t> sensorsArray;
  sensorsArray.reserve(numSens9);
  /**
   * Create a variable for the speed command (motor_command).
   * This will be received from the output of NN and is used to adjust the speed of motors.
   * There are also 3 additional variables which could be used for sending different 
   * speed commands to the arduino in the future. At the moment these are all equal to
   * the speed command (i.e. the output of NN)
   **/
  int motor_command = 0;
  int16_t left_velocity = (int16_t)0;
  int16_t right_velocity = (int16_t)0;
  int16_t differential_velocity = (int16_t)0;
  /**
   * This is an infinite loop.
   * It stops if there is an error occures or we exit the program
   **/
  for (;;){
    /**
     * Creating a window for GUI
     **/
    statFrame = cv::Scalar(100, 100, 100);
    // # SECTION: CAMERA INPUTS
    /**
     * Creating two matrix to capture camera view.
     **/  
    Mat origframe, frame;
    /**
     * Put the raw camera view into the 'origframe' matrix.
     **/ 
    cap >> origframe;
    /**
     * Flip the camera view if needed: 0 horizontal, 1 vertical, -1 both.
     * Save the new view into the 'frame' matrix.
     **/
    flip(origframe,frame,-1);
    /**
     * Clear the predictors vector from its previous values.
     **/
    predictorDeltaMeans.clear();
    /**
     * Call the 'calcPredictors' function from the exernal class we created above.
     * Pass the camera view 'frame' to extract the predictors from.
     * Pass a pointer to an empty vector 'predictorDeltaMeans'
     * where the new predictor values will be stored.
     * It returns the angle of deviation, this is not used for the normal Back-Propagation learning
     **/
    external->calcPredictors(frame, predictorDeltaMeans);

    // # SECTION: SENSOR INPUTS
    /**
     * Clear the vector from its previous values
     **/
    sensorsArray.clear();
    /**
     * Create an array to read the raw sensor data in
     **/
    uint8_t readToThis[numSens9] = {0,0,0,0,0,0,0,0,0};
    /**
     * Use the seriallis 'LS' to read the sesnor values
     * It returns 1 if successful, otherwise the error will stop the program as explained above
     **/
    Ret = LS.Read(&readToThis, sizeof(readToThis));
    /**
     * Unpacking the sensor inputs into the vector created above
     **/
    for (int i = 0 ; i < numSens9; i++){
      sensorsArray.push_back(readToThis[i]);
    }

    // # SECTION: THE REFLEX ERROR
    /**
     * Call 'calcError' function from the external class to calculate the error
     * Pass the GUI 'stateFrame' where the error will be plotted
     * Pass the sensor values 'sensorsArray' that were obtained from the arduino
     * It returns the reflex error 'reflex_error' from the sensors 
     * This is the reflex error that is used to adjust the speed of the motors
     * This is also used as the feedback that trains the NN
     **/
    double reflex_error = external->calcError(statFrame, sensorsArray);

    // # SECTION: MOTOR COMMAND
    if (Ret > 0){ // if Ret is 1 means there is no errors
    /**
     * If the reflex error 'reflex_error' is greater than 99
     * it sends a value of +-19 to each motors which stops the robot
     * When the learning has succeeded or a certain number of steps is completed.
     * the reflex error is manually set to 100 (in external.cpp file)
     * which satisfies the if-statement below.
     * This stops the robot and then the program terminates.
     * This is to stop extra unwanted data being written to the files
     * and also to stop the robot from falling off the table after the program has ended.
     **/
      if(abs(reflex_error)  > 99){
        left_velocity = (int16_t)(0-19);
        right_velocity = (int16_t)19;
      }else{ // If the error is less than 100 the robot continues.
        /**
         * The reflex_error is not the only driver of the motors
         * It joins with the output of the network to drive the robot.
         * On each completed step it calls the 'onStepCompleted' function
         * from the external class which runs the NN and generates the NN output
         * This output is then summed with reflex_error (a weighted sum)
         * to drive the navigation.
         * Pass the 'statFrame' where the output of NN will be displayed
         * Pass the 'reflex_error' to train the NN and to drive the robot together with NN's output
         * Pass the 'angle of deviation', this is for another learning paradigm (PaM).
         * Pass the 'predictorDeltaMeans' for the NN to predict the future actions
         * It returns the 'motor_command' which is used to drive the motors
         **/
        motor_command = external->onStepCompleted(statFrame, reflex_error, predictorDeltaMeans); //******************************************ADDITION********************
        /**
         * The differential, left and right velocities are all equal to the motor_command
         * In the future they could be used to send different values to the motors
         **/
        differential_velocity = (int16_t)motor_command;
        left_velocity = (int16_t)motor_command;
        right_velocity = (int16_t)motor_command;
      }
      /**
       * Sending a start marker to the robot to synchoronise the communication
       **/
      int16_t startMarker = 32767;
      /**
       * Putting all 4 variables in one array to send to arduino
       **/
      int16_t motor_array_command[4] = {differential_velocity, left_velocity , right_velocity, startMarker};
      /**
       * Sending the motor command to the arduino
       **/
      Ret = LS.Write(&motor_array_command, sizeof(motor_array_command));
  }
    // SECTION: GUI UPDATE
    cvui::update(); // update the windows
    cv::imshow(STAT_WINDOW, statFrame); // replot and show the frame
    /**
     * If the reflex is greater than 99 (= 100) it breaks the if-statement 
     * which stops the program and returns 0
     **/
    if(abs(reflex_error) > 99){
      break;
    }
    /**
     * If the Esc key is pressed it breaks the if-statement
     * Which stops the program and returns 0
     **/
    if (waitKey(20) == ESC_key)
      break;
  }
  return 0;
}
