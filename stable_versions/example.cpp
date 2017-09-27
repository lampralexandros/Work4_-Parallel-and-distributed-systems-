/**********************************************************************************************************
  DEMO CODE: XOR Backpropagation Example
  File: example.cpp
  Version: 0.1
  Copyright(C) NeuroAI (http://www.learnartificialneuralnetworks.com)
  Documentation: http://www.learnartificialneuralnetworks.com/neural-network-software/backpropagation-source-code/
  NeuroAI Licence:
  Permision granted to use this source code only for non commercial and educational purposes.
  You can distribute this file but you can't take ownership of it or modify it.
  You can include this file as a link on any website but you must link directly to NeuroAI website
  (http://www.learnartificialneuralnetworks.com)
  Written by Daniel Rios <daniel.rios@learnartificialneuralnetworks.com> , June 2013

 /*********************************************************************************************************/

#include <iostream>
#include <cstdio>
#include <ctime>
#include "bpnet.h"
using namespace std;
#define PATTERN_COUNT 4
#define PATTERN_SIZE 2
#define NETWORK_INPUTNEURONS 3
#define NETWORK_OUTPUT 1
#define HIDDEN_LAYERS 1
#define EPOCHS 1000000



int main()
{
    clock_t start;
    double duration;



    /* Your algorithm here */


    //Create some patterns
    //playing with xor
    //XOR input values
    float pattern[PATTERN_COUNT][PATTERN_SIZE]=
    {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };

    //XOR desired output values
    float desiredout[PATTERN_COUNT][NETWORK_OUTPUT]=
    {
        {0},
        {1},
        {1},
        {0}
    };


    bpnet net,net1;//Our neural network object
    int i,j;
    float error;
    //We create the network

    int hiddenlayercount=HIDDEN_LAYERS;
    int hiddenlayerNeuronCount[HIDDEN_LAYERS]={2}; //array, each element represents how many neurons each hidden layer has
    net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,hiddenlayercount);
    //net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,HIDDEN_LAYERS,HIDDEN_LAYERS);

    net1.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,hiddenlayercount);

    net1.clone_bpnet(&net);


    //Start the neural network training
    start = clock();
    cout << "Start training for " << EPOCHS << " " << endl;
    int counter=0;
    for(i=0;i<EPOCHS;i++)
    {
        error=0;
        for(j=0;j<PATTERN_COUNT;j++)
        {

            error+=net1.train(desiredout[j],pattern[j],0.2f,0.1f,1);
            counter++;
            if(counter==10){
              net1.applyBatchCumulations(0.2f,0.1f);
              counter=0;
            }




        }


        error/=PATTERN_COUNT;
        //display error
        //cout << "ERROR:" << error << "\r";

    }
    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "Train net duration : " << duration << endl;
    //once trained test all patterns
    for(i=0;i<PATTERN_COUNT;i++)
    {

        net1.propagate(pattern[i]);

    //display result
        cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << *desiredout[i] << " NET RESULT: "<< net1.getOutput().neurons[0]->output << endl;
    }

    return 0;
}
