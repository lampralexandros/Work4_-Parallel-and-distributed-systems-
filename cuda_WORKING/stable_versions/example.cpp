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
#include <cuda.h>

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
        printf("hidden layer neuron count =%d \n",hiddenlayerNeuronCount[i]);
      }
    }else{
        printf("Enter number of neurons every hidden layer will have\n");
        hiddenlayerNeuronCount=(int *)malloc(HIDDEN_LAYERS*sizeof(int));
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




    clock_t start,start1;
    double duration,duration1;//,dataduration;





    /* Your algorithm here */


    //Create some patterns
    //playing with xor
    //XOR input values



    float pattern[PATTERN_COUNT][PATTERN_SIZE]=
    {//I XOR I XOR  I
          // {0,0,0},
          // {0,0,1},
          // {0,1,0},
          // {0,1,1},
          // {1,0,0},
          // {1,0,1},
          // {1,1,0},
          // {1,1,1}
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
    // desiredout[4][0]=1;
    // desiredout[5][0]=0;
    // desiredout[6][0]=0;
    // desiredout[7][0]=1;


    // desiredout[0][0]=0;
    // desiredout[1][0]=1;
    // desiredout[2][0]=1;
    // desiredout[3][0]=0;

    bpnet *netMatrix=new bpnet[2];//Our neural network object
    int i,j;
    //float error;
    //We create the network



    netMatrix[0].create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,HIDDEN_LAYERS);
    netMatrix[1].create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,HIDDEN_LAYERS);
    netMatrix[1].clone_bpnet(&netMatrix[0]);

    int max_neurons=netMatrix[0].max_neuroncount();
    int max_inputs=netMatrix[0].max_inputcount();

    //net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,HIDDEN_LAYERS,HIDDEN_LAYERS);

    //Start the neural network training
    start = clock();
    cout << "Start training for " << EPOCHS*PATTERN_COUNT << " " << endl;
    int totalbatchescounter=0;
    int totalcounter=0;
    int counter=0;
    duration1=0;
    // dataduration=0;
    for(i=0;i<EPOCHS;i++)//EPOCHS
    {
        //error=0;
        for(j=0;j<PATTERN_COUNT;j++)
        {
            totalcounter++;
            netMatrix[0].batchTrain(desiredout[j],pattern[j]);


            //train from 2ond net
            counter++;
            if(counter==10){
              //function to add errors
               totalbatchescounter+=1;
               start1 = clock();
               netMatrix[0].KernelapplyBatchCumulations(0.2f,0.1f,max_neurons,max_inputs);
               //netMatrix[0].applyBatchCumulations(0.2f,0.1f);
               duration1 += ( clock() - start1 ) / (double) CLOCKS_PER_SEC;


              counter=0;
            }

        }



    }
    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "Train batch cummulations duration : " << duration1 << endl;
    cout << "Total TRAINS trained :"<<totalcounter<<endl;
    cout << "Total batches trained :"<<totalbatchescounter<<endl;
      // cout << "Total data transfer duration:"<<dataduration<<endl;

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
