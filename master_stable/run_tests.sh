# A simple bash script that for running experiments
# Note: To run the script make sure it you have execution rights 
# (use: chmod u+x run_tests.sh to give execution rights) 
!/bin/bash

#NAME="backpropagation"
#DATE=$(date "+%Y-%m-%d-%H:%M:%S")
#FILE_PREF=$NAME-pthread

#echo $NAME
#echo $DATE

make clean; make

# run Neural Network experiments
for PATTERN_SIZE in 2 3 4 5 ; do \
    for HIDDEN_LAYERS in 2 ; do \
        for NEURONS in 2 ; do \
		for EPOCHS in 10000; do \
			for NUM_THREADS in 2 4; do \
				for SELECT in 2; do \
					for EXPORT_FILE in 2; do \

					     	./backpropagation $PATTERN_SIZE $HIDDEN_LAYERS $NEURONS $EPOCHS $NUM_THREADS $SELECT $EXPORT_FILE
						./backpropagationserial $PATTERN_SIZE $HIDDEN_LAYERS $NEURONS $EPOCHS $NUM_THREADS 1 $EXPORT_FILE

					done ; \
				done ; \
    			done ; \
		done ; \
	done ; \
    done ; \
done ;

for PATTERN_SIZE in 5 ; do \
    for HIDDEN_LAYERS in 2 ; do \
        for NEURONS in 2 3 4 5 6 7 8 9 ; do \
		for EPOCHS in 10000; do \
			for NUM_THREADS in 2 4; do \
				for SELECT in 2; do \
					for EXPORT_FILE in 1; do \

					     	./backpropagation $PATTERN_SIZE $HIDDEN_LAYERS $NEURONS $EPOCHS $NUM_THREADS $SELECT $EXPORT_FILE
						./backpropagationserial $PATTERN_SIZE $HIDDEN_LAYERS $NEURONS $EPOCHS $NUM_THREADS 1 $EXPORT_FILE
					done ; \
				done ; \
    			done ; \
		done ; \
	done ; \
    done ; \
done ;

mv *.txt results/


