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
    // adding a int to by pass the error of HIDDEN_LAYERS
    int i,j,hidden_layers_var;
    float error;
    hidden_layers_var=HIDDEN_LAYERS;
    //We create the network
    net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,&hidden_layers_var,HIDDEN_LAYERS);

    //tests
    printf("This is  a test %d\n",net.get_m_hiddenlayercount() );
      hidden_layers_var=2;
    net1.create(3,4,NETWORK_OUTPUT,&hidden_layers_var,2);
    printf("This is  a 2test %d\n",net1.get_m_hiddenlayercount() );
    net1.clone_bpnet(&net);
    printf("This is  a 3test %d\n",net1.get_m_hiddenlayercount() );

    printf("This is  a 4test %d\n",net.get_m_hiddenlayercount() );

    //Start the neural network training
    start = clock();
    cout << "Start training for " << EPOCHS << " " << endl;
    for(i=0;i<EPOCHS;i++)
    {
        error=0;
        for(j=0;j<PATTERN_COUNT;j++)
        {
            error+=net.train(desiredout[j],pattern[j],0.2f,0.1f);
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

        net.propagate(pattern[i]);

    //display result
        cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << *desiredout[i] << " NET RESULT: "<< net.getOutput().neurons[0]->output << endl;
    }

    return 0;
}
