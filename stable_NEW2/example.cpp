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
#include <pthread.h>

using namespace std;
// #define PATTERN_COUNT 4
// #define PATTERN_SIZE 2
#include "threadpool/ThreadPool.h"

// ThreadPool and Mutex
pthread_mutex_t mutexBusy;
static int mutexsum;
static int *mutextable;

// Net matrix as global so we can access it with out data_wraper
static bpnet *netMatrix;
static float **pattern;
static float **desiredout;
static int main_start_batch;
static int threads_start_batch;



struct data_wraper{
  int pattern_size,counter,nnet_id;

  data_wraper(){}
  ~data_wraper(){}

};

// not needed
const int MAX_TASKS = 4;

void train_wrapper(void* arg)
{

  data_wraper* data1 = (data_wraper*) arg;
  int i=data1->nnet_id*data1->counter;

  while(data1->counter >0){
    netMatrix[data1->nnet_id].batchTrain(desiredout[i],pattern[i]);
    i++;
    data1->counter--;
  }

//TODELETE/  cout<<"train3"<<"threadid="<<data1->nnet_id<<endl;
//Transfering start
if((data1->nnet_id)==0){
main_start_batch=i;
}
//TODELETE/cout<<"train4"<<"threadid="<<data1->nnet_id<<endl;
  pthread_mutex_lock (&mutexBusy);
//TODELETE/cout<<"train4.1"<<"threadid="<<data1->nnet_id<<endl;
  mutexsum+=-1;
//TODELETE/cout<<"train4.2"<<"threadid="<<data1->nnet_id<<endl;
  if(data1->nnet_id!=0){
    mutextable[(data1->nnet_id)-1]=1;
  }
//TODELETE/cout<<"train4.3"<<"threadid="<<data1->nnet_id<<endl;
  pthread_mutex_unlock (&mutexBusy);
//TODELETE/cout<<"train5"<<"threadid="<<data1->nnet_id<<endl;
}

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

  NETWORK_INPUTNEURONS=PATTERN_SIZE;
  PATTERN_COUNT=1<<PATTERN_SIZE;

    clock_t start_clock;
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

  //cout << "Ccreating nnet = number of workers:"<<NUM_THREADS <<"Batch size="<<batch_size<< endl;
  //TODELET/bpnet *netMatrix=new bpnet[NUM_THREADS];//Our neural network object
  netMatrix=new bpnet[NUM_THREADS];
  cout << "Cloning the net maybe Completed l253"<< endl;
  netMatrix[0].create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,HIDDEN_LAYERS);
  for(int i=1;i<NUM_THREADS;i++){
  netMatrix[i].create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hiddenlayerNeuronCount,HIDDEN_LAYERS);
  netMatrix[i].clone_bpnet(&netMatrix[0]);
  }

  //CHECKING FOR CUDA ERRORS
  int max_neurons=netMatrix[0].max_neuroncount();
  int max_inputs=netMatrix[0].max_inputcount();
  int number_of_layers=2+HIDDEN_LAYERS;
  int exitflag=1;
  for(int i=32;i>1;i--){
    if(((max_neurons*number_of_layers)%i)==0){
      exitflag=0;
      break;
    }
  }
  if(exitflag==0){
    for(int i=32;i>1;i--){
      if(((max_neurons*max_neurons*number_of_layers)%i)==0){
        exitflag=2;
        break;
      }
    }
  }
  if(exitflag==0){
    exitflag=1;
  }

  if(exitflag==1){
    printf("ERROR:\nCan't find proper grid for Cuda\nmodulo[max_neurons*number_of_layers,32:-1:2] must be 0 at least once\nExiting programm\n");
    exit(1);
  }





  int batch_per_thread=PATTERN_COUNT/NUM_THREADS;
  cout<<"Each thread must train "<<batch_per_thread<<" patterns"<<endl;
    // if((batch_size%NUM_THREADS==0) && batch_size>NUM_THREADS){
    //   cout << "Calculating batch_size="<<batch_size <<" / workers="<< NUM_THREADS <<" == "<<batch_per_thread<<endl;
    // }
    // else{
    //   cout << "Calculating batch_size="<<batch_size <<" / workers="<< NUM_THREADS <<" == "<<batch_per_thread<<endl;
    //   cout<<"EXITING"<<endl;
    //   exit(1);
    // }

    //Init Thread Pool
    ThreadPool tp(NUM_THREADS-1);
    if(NUM_THREADS-1>0){
    int ret = tp.initialize_threadpool();
    if (ret == -1) {
      cerr << "Failed to initialize thread pool!" << endl;
      return 0;
      }
    }
    // init mutex to join
    pthread_mutex_init(&mutexBusy, NULL);
    mutexsum=0;
    mutextable =(int *)malloc(sizeof(int)*((NUM_THREADS)-1));

    //Start the neural network training
    start_clock = clock();
    cout << "Start training for " << EPOCHS << " " << endl;
    data_wraper data2;
    data2.nnet_id=0;
    data2.counter=batch_per_thread;
    data2.pattern_size=PATTERN_COUNT;
    main_start_batch=0;
    threads_start_batch=0;
    //data2=(data_wraper *)malloc(sizeof(data_wraper)*NUM_THREADS);

    // for(int i=0;i<NUM_THREADS;i++){
    //   data2[i].start=(int *)malloc(sizeof(int));
    // }


    int flag=1;

    for(int i=0;i<EPOCHS;i++)
    {
        pthread_mutex_lock (&mutexBusy);
        // reinitiallizing for synch method
        mutexsum=NUM_THREADS;
        //reinitiallizing for variable to cover such cost with gatherErrors2
        for(int j=0;j<NUM_THREADS-1;j++){
          mutextable[j]=0;
        }
        pthread_mutex_unlock (&mutexBusy);

        // Loop to send out work to thread pool
        for(int j=1;j<NUM_THREADS;j++)
        {
          // data2[j].nnet_id=j;
          // data2[j].counter=batch_per_thread;
          // data2[j].pattern_size=PATTERN_COUNT;
          //TODELETE/cout<<"Inside epoch="<<i<<"thread="<<j<<endl;
          data_wraper *data;
          data=new data_wraper();
          data->nnet_id=j;
          data->counter=batch_per_thread;
          data->pattern_size=PATTERN_COUNT;
          // data2[j].pattern_size=PATTERN_COUNT;
          //train_wrapper((void *) data);
          if(j!=0){
            //Task* t = new Task(&train_wrapper, (void*) &data2[j]);
            Task* t = new Task(&train_wrapper, (void*) data);
            tp.add_task(t);
          }
          //TODELETE/cout<<"Inside2 epoch="<<i<<"thread="<<j<<endl;
        }
        //TODELETE/cout<<"Outside epoch="<<i<<endl;
        data2.counter=batch_per_thread;
        train_wrapper((void *) &data2);
        //Synch method
        flag=1;
        while(flag)
        {
          if(pthread_mutex_trylock(&mutexBusy)==0)
          {
            if(mutexsum==0){
              flag=0;
              }
            pthread_mutex_unlock (&mutexBusy);
            for(int j=0;j<NUM_THREADS-1;j++){
              if(mutextable[j]==1){
                netMatrix[0].gatherErrors2(netMatrix,j+1);
                mutextable[j]=2;
              }
            }

            }
          }
      // // transfering new start to threads
      threads_start_batch=main_start_batch;
      // Finnishing gatherErrors2
      for(int j=0;j<NUM_THREADS-1;j++){
          if(mutextable[j]==1){
            //TODELETE/cout<<"Never inside"<<endl;
            netMatrix[0].gatherErrors2(netMatrix,j+1);
            mutextable[j]=2;
          }
        }

        netMatrix[0].applyBatchCumulations(0.2f,0.1f);
        //netMatrix[0].KernelapplyBatchCumulations(0.2f,0.1f,max_neurons,max_inputs);
        for(int j=1;j<NUM_THREADS;j++){
        netMatrix[j].clone_bpnet(&netMatrix[0]);
        }
        //TODELETE/cout<<"Finishing epoch="<<i<<endl;
    }


    duration = ( clock() - start_clock ) / (double) CLOCKS_PER_SEC;
    cout << "Train net duration : " << duration/60 <<" mins"<< endl;
    //once trained test all patterns
    float square_error=0;
    for(int i=0;i<PATTERN_COUNT;i++)
    {
        netMatrix[0].propagate(pattern[i]);
    //display result
        cout << "TESTED PATTERN " << i << " DESIRED OUTPUT: " << *desiredout[i] << " NET RESULT: "<< netMatrix[0].getOutput().neurons[0]->output << endl;
        square_error+=(*desiredout[i]-netMatrix[0].getOutput().neurons[0]->output)*(*desiredout[i]-netMatrix[0].getOutput().neurons[0]->output);
    }
    square_error/=PATTERN_COUNT;
    cout<<"square_error= "<<square_error<<endl;
    tp.destroy_threadpool();
    return 0;
}
