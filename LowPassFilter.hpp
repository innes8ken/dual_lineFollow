
#ifndef _LowPassFilter_hpp_
#define _LowPassFilter_hpp_

#include <iostream>
#include <cmath>


/**
 * @brief Allows creation of low-pass filters 
 * 
 */
class LowPassFilter{
public:
	//constructors
	LowPassFilter();

	/**
	 * @brief Construct a new Low Pass Filter object
	 * 
	 * @param iCutOffFrequency takes the desired cut off frequency 
	 */
	LowPassFilter(float iCutOffFrequency);

	/**
	 * @brief Construct a new Low Pass Filter object
	 * 
	 * @param iCutOffFrequency takes the desired cut off frequency
	 * @param iDeltaTime desired time change ?
	 */

	LowPassFilter(float iCutOffFrequency, float iDeltaTime);
	//functions

	/**
	 * @brief update the filtered output 
	 * 
	 * @param input 
	 * @return float output
	 */
	float update(float input);

	/**
	 * @brief update the filtered output
	 * 
	 * @param input 
	 * @param deltaTime re-set deltatime 
	 * @return float  output
	 */
	float update(float input, float deltaTime);
	//get and set funtions
	
	float getOutput();
	float getCutOffFrequency();
	void setCutOffFrequency(float input);
	void setDeltaTime(float input);
private:
	float output = 0;
	float cutOffFrequency = 0;
	float ePow = 0;
};

#endif //_LowPassFilter_hpp_
