# installing boost
	# to be completed
# running test to check boost main
echo "Compile to check boost.h"
g++ -w -std=gnu99  test_boost.cpp -lm -I /usr/local/boost_1_65_1/boost -o test_boost
echo "Running Test should get 1 2 3 * 3 = 3 6 9"
echo 1 2 3 | ./test_boost
echo "done removing test_boost"
rm test_boost
echo "Compile to test thread pool"
g++ -w -std=gnu99  test_boost_threads.cpp -lm -I /usr/local/boost_1_65_1 -o test_boost -L /usr/local/boost_1_65_1/stage/lib
echo "Running Test"
./test_boost
