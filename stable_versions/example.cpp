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
  int *hiddenlayerNeuronCount;

  if (argc < 6) { // Check if the command line arguments are correct

    NETWORK_INPUTNEURONS=3;
    NETWORK_OUTPUT=1;
    HIDDEN_LAYERS=2;
    hiddenlayerNeuronCount=(int *)malloc(HIDDEN_LAYERS*sizeof(int));
    hiddenlayerNeuronCount[0]=3;
    hiddenlayerNeuronCount[1]=3;
    EPOCHS=1000000;
    NUM_THREADS=4;
    printf("Usage: %s NI NO HI EPOCHS THREADS\n\n", argv[0]);

    printf("Using default net configurations \n\n"
	   "  NI     : number of inputs= %d\n"
     "  NO     : number of outputs= %d\n"
	   "  HI     : numer of hiddenlayers= %d\n"
     "  hiddenlayerNeuronCount   : number of hidenlayer neurons= %d\n"
     "  EPOCHS : number of EPOCHS= %d\n"
     "  NUM_THREADS : number of THREADS= %d\n\n"
     ,NETWORK_INPUTNEURONS, NETWORK_OUTPUT,HIDDEN_LAYERS,hiddenlayerNeuronCount[0],EPOCHS,NUM_THREADS);





  }else if (argc==6){
    NETWORK_INPUTNEURONS = atoi(argv[1]);        // Rows
    NETWORK_OUTPUT = atoi(argv[2]);       //Columns
    HIDDEN_LAYERS = atoi(argv[3]); // Propability of life cell
    EPOCHS = atoi(argv[4]);     // Display output
    NUM_THREADS=atoi(argv[5]); //number of threads to use

    hiddenlayerNeuronCount=(int *)malloc(HIDDEN_LAYERS*sizeof(int));
    printf("Do you want all hiddenlayers to have same numer of neurons?YES 1, NO 0\n");
    int choice;

    cin>>choice;
    if(choice==0){
      for(int i=0;i<HIDDEN_LAYERS;i++){
        printf("Enter number of neurons for %d hidden layer\n",(i+1));
        int number_of_neurons;
        number_of_neurons=getchar();
        hiddenlayerNeuronCount[i]=number_of_neurons;
        printf("hidden layer neuron count = \n",hiddenlayerNeuronCount[i]);
      }
    }else{
        printf("Enter number of neurons every hidden layer will have\n");
        int number_of_neurons;
        cin >> number_of_neurons;
        cout<< "number_of_neurons="<< number_of_neurons<< endl;
        for(int i=0;i<HIDDEN_LAYERS;i++){
          hiddenlayerNeuronCount[i]=number_of_neurons;
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
    /*float desiredout[PATTERN_COUNT][NETWORK_OUTPUT]=
    {
        {0},
        {1},
        {1},
        {0}
    };
    */
    float **desiredout;
    desiredout=(float **)malloc(PATTERN_COUNT*(sizeof(float *)));
    for(int i=0;i<PATTERN_COUNT;i++){
      desiredout[i]=(float *)malloc(NETWORK_OUTPUT*(sizeof(float)));
    }

    desiredout[0][0]=0;
    desiredout[1][0]=1;
    desiredout[2][0]=1;
    desiredout[3][0]=0;

    bpnet *netMatrix=new bpnet[2];//Our neural network object
    int i,j;
    float error;
    //We create the network



    netMatrix[0].create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,HIDDEN_LAYERS);
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

            netMatrix[0].batchTrain(desiredout[j],pattern[j]);
            //train from 2ond net
            counter++;
            if(counter==10){
              //function to add errors
              netMatrix[0].gatherErrors(netMatrix,2);
              netMatrix[0].applyBatchCumulations(0.2f,0.1f);
              counter=0;
            }

        }



    }
    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "Train net duration : " << duration << endl;
    //once trained test all patterns

    for(i=0;i<PATTERN_COUNT;i++)
    {

        netMatrix[0].propagate(pattern[i]);

    //display result
        cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << *desiredout[i] << " NET RESULT: "<< netMatrix[0].getOutput().neurons[0]->output << endl;
    }

    return 0;
}
