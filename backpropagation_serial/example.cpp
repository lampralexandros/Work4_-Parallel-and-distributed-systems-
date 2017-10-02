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

static float **pattern;
static float **desiredout;
// #define PATTERN_COUNT 4
// #define PATTERN_SIZE 2




int main(int argc, char *argv[])
{
  int NETWORK_INPUTNEURONS,NETWORK_OUTPUT,HIDDEN_LAYERS,EPOCHS,NUM_THREADS,PATTERN_SIZE,PATTERN_COUNT;
  NETWORK_OUTPUT=1;
  int *hiddenlayerNeuronCount;
  //int batch_size=20;


  if (argc < 5) { // Check if the command line arguments are correct

    PATTERN_SIZE=3;
    HIDDEN_LAYERS=2;
    hiddenlayerNeuronCount=(int *)malloc(HIDDEN_LAYERS*sizeof(int));
    hiddenlayerNeuronCount[0]=3;
    hiddenlayerNeuronCount[1]=3;
    EPOCHS=100000;
    NUM_THREADS=4;
    //batch_size=20;

    printf("Usage: %s NI NO HI EPOCHS THREADS\n\n", argv[0]);

    printf("Using default net configurations \n\n"
	   "  PATTERN_SIZE     : number of inputs= %d\n"
	   "  HI     : numer of hiddenlayers= %d\n"
     "  hiddenlayerNeuronCount   : number of hidenlayer neurons= %d\n"
     "  EPOCHS : number of EPOCHS= %d\n"
     "  NUM_THREADS : number of THREADS= %d\n\n"
     ,PATTERN_SIZE,HIDDEN_LAYERS,hiddenlayerNeuronCount[0],EPOCHS,NUM_THREADS);





  }else if (argc==5){
    PATTERN_SIZE = atoi(argv[1]);        // Rows
    HIDDEN_LAYERS = atoi(argv[2]);       //Columns
    EPOCHS = atoi(argv[3]); // Propability of life cell
    NUM_THREADS = atoi(argv[4]);     // Display output
    // batch_size=atoi(argv[5]);
    //NUM_THREADS=atoi(argv[5]); //number of threads to use
    hiddenlayerNeuronCount=(int *)malloc(HIDDEN_LAYERS*sizeof(int));
    printf("Do you want all hiddenlayers to have same numer of neurons?YES 1, NO 0\n");
    int choice;
    cin>>choice;
    if(choice==0){
      for(int i=0;i<HIDDEN_LAYERS;i++){
        printf("Enter number of neurons for %d hidden layer\n",(i+1));
        int number_of_neurons;
        cin >> number_of_neurons;
        hiddenlayerNeuronCount[i]=number_of_neurons;
        printf("HIDDEN_LAYERS[%d] has %d neurons \n",i,hiddenlayerNeuronCount[i]);
      }
    }else{
        printf("Enter number of neurons every hidden layer will have\n");
        int number_of_neurons;
        cin >> number_of_neurons;
        cout<< "HIDDEN_LAYERS["<<HIDDEN_LAYERS<<"]="<< number_of_neurons<< endl;
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

  NETWORK_INPUTNEURONS=PATTERN_SIZE;
  PATTERN_COUNT=1<<PATTERN_SIZE;
  cout<<"Number of patterns produced with "<<PATTERN_SIZE<<" is "<<PATTERN_COUNT<<endl;



    clock_t start;
    double duration;



    pattern=(float **)malloc(PATTERN_COUNT*(sizeof(float *)));
    for(int i=0;i<PATTERN_COUNT;i++){
      pattern[i]=(float *)malloc(PATTERN_SIZE*(sizeof(float)));
    }


    for(int i=0;i<(1<<PATTERN_SIZE);i++){
      for(int j=0;j<PATTERN_SIZE;j++){
           pattern[i][j]=i/(1<<(j))%2;
           //cout <<" "<<pattern[i][j]<<" ";
      }
    }


    desiredout=(float **)malloc(PATTERN_COUNT*(sizeof(float *)));
      for(int i=0;i<(1<<PATTERN_SIZE);i++){
        desiredout[i]=(float *)malloc(NETWORK_OUTPUT*(sizeof(float)));
      }


      for(int i=0;i<(1<<PATTERN_SIZE);i++){
        desiredout[i][0]= pattern[i][0];
        for(int j=1;j<PATTERN_SIZE;j++){
            desiredout[i][0]= (int)desiredout[i][0] ^(int)pattern[i][j];
        }
      }

    //bpnet net;
    bpnet net;

    //We create the network


    //net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,HIDDEN_LAYERS);
    net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,HIDDEN_LAYERS);


    //net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,HIDDEN_LAYERS,HIDDEN_LAYERS);

    //Start the neural network training
    start = clock();
    cout << "Start training for " << EPOCHS << " " << endl;
    int batch_size=100;
    int randomPattern;
    for(int i=0;i<EPOCHS;i++)
    {
        for(int j=0;j<PATTERN_COUNT;j++)
        {

            //cout<<"random= "<<randomPattern<<endl;
            net.batchTrain(desiredout[j],pattern[j]);

        }
        // randomPattern = rand()%(PATTERN_COUNT-0) + 0;
        // net.batchTrain(desiredout[randomPattern],pattern[randomPattern]);

          net.applyBatchCumulations(0.2f,0.1f);
          if(i%10000==0){
            cout<<"\r"<<"Current EPOCH = "<<i<<flush;
          }
    }
    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout<<endl;
    cout << "Train net duration : " << duration/60<<" mins" << endl;
    //once trained test all patterns
    float square_error=0;
    for(int i=0;i<PATTERN_COUNT;i++)
    {

        net.propagate(pattern[i]);

    //display result
    square_error+=(*desiredout[i]-net.getOutput().neurons[0]->output)*(*desiredout[i]-net.getOutput().neurons[0]->output);
        cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << *desiredout[i] << " NET RESULT: "<< net.getOutput().neurons[0]->output << endl;
    }
    square_error/=PATTERN_COUNT;
  cout<<"square_error= "<<square_error<<endl;

    return 0;
}
