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
#include <stdlib.h>
#include <ctime>
#include "bpnet.h"
using namespace std;
#define PATTERN_COUNT 4
#define PATTERN_SIZE 2




int main(int argc, char *argv[])
{
  int NETWORK_INPUTNEURONS,NETWORK_OUTPUT,HIDDEN_LAYERS,EPOCHS,NUM_THREADS;
  int *HI_C;

  if (argc < 6) { // Check if the command line arguments are correct

    NETWORK_INPUTNEURONS=3;
    NETWORK_OUTPUT=1;
    HIDDEN_LAYERS=2;
    HI_C=(int *)malloc(HIDDEN_LAYERS*sizeof(int));
    HI_C[0]=3;
    HI_C[1]=3;
    EPOCHS=1000000;
    NUM_THREADS=4;
    printf("Usage: %s NI NO HI EPOCHS THREADS\n\n", argv[0]);

    printf("Using default net configurations \n\n"
	   "  NI     : number of inputs= %d\n"
     "  NO     : number of outputs= %d\n"
	   "  HI     : numer of hiddenlayers= %d\n"
     "  HI_C   : number of hidenlayer neurons= %d\n"
     "  EPOCHS : number of EPOCHS= %d\n"
     "  NUM_THREADS : number of THREADS= %d\n\n"
     ,NETWORK_INPUTNEURONS, NETWORK_OUTPUT,HIDDEN_LAYERS,HI_C[0],EPOCHS,NUM_THREADS);





  }else if (argc==6){
    if(atoi(argv[1])!=3)
    NETWORK_INPUTNEURONS = atoi(argv[1]);        // Rows
    NETWORK_OUTPUT = atoi(argv[2]);       //Columns
    HIDDEN_LAYERS = atoi(argv[3]); // Propability of life cell
    EPOCHS = atoi(argv[4]);     // Display output
    NUM_THREADS=atoi(argv[5]); //number of threads to use

    HI_C=(int *)malloc(HIDDEN_LAYERS*sizeof(int));
    printf("Do you want all hiddenlayers to have same numer of neurons?YES 1, NO 0\n");
    int choice;
    int number_of_neurons;
    cin>>choice;
    if(choice==0){
      for(int i=0;i<HIDDEN_LAYERS;i++){
        printf("Enter number of neurons for %d hidden layer\n",(i+1));
        cin>>number_of_neurons;
        HI_C[i]=number_of_neurons;
        printf("hidden layer neuron count = \n",HI_C[i]);
      }
    }else{
        printf("Enter number of neurons every hidden layer will have\n");

        cin>>number_of_neurons;
        for(int i=0;i<HIDDEN_LAYERS;i++){
          HI_C[i]=number_of_neurons;
        }


    }


  }else{
    printf("Usage: %s NI NO HI EPOCHS THREADS\n"
	   "where\n"
	   "  NI     : number of inputs\n"
     "  NO     : number of outputs\n"
	   "  HI     : numer of hiddenlayers\n"
     "  EPOCHS : number of EPOCHS\n"
     "  THREADS : number of THREADS\n"
     , argv[0]);

     printf("Exiting programm \n");
     return(1);

  }




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


    bpnet net;//Our neural network object
    int i,j;
    float error;
    //We create the network

    int *hiddenlayerNeuronCount=new int[HIDDEN_LAYERS]; //array, each element represents how many neurons each hidden layer has

    for(int i=0;i<HIDDEN_LAYERS;i++){
      hiddenlayerNeuronCount[i]=HI_C[i];
    }


    net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,HIDDEN_LAYERS);
    //net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,HIDDEN_LAYERS,HIDDEN_LAYERS);

    //Start the neural network training
    start = clock();
    cout << "Start training for " << EPOCHS << " " << endl;
    int counter=0;

    for(i=0;i<EPOCHS;i++)
    {
        error=0;
        for(j=0;j<PATTERN_COUNT;j++)
        {

            error+=net.train(desiredout[j],pattern[j],0.2f,0.1f,1);
            counter++;
            if(counter==10){
              net.applyBatchCumulations(0.2f,0.1f);
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

        net.propagate(pattern[i]);

    //display result
        cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << *desiredout[i] << " NET RESULT: "<< net.getOutput().neurons[0]->output << endl;
    }

    return 0;
}
