#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

void inputFromKeyboard(int argc, char* argv[], double *beginTemp, double *endTemp,
                            double *numOfGrid, double *numOfTimeStamp) {
    *beginTemp = atof(argv[1]);
    *endTemp = atof(argv[2]);
    *numOfGrid = atof(argv[3]);
    *numOfTimeStamp = atof(argv[4]);
}

int main(int argc, char *argv[]) {
	int numOfProcess;
    int numOfRank;
    double beginTemp;
    double endTemp;
    double numOfGrid;
    double numOfTimeStamp;

    MPI_Init(&argc, &argv);
    inputFromKeyboard(argc, argv, &beginTemp, &endTemp, &numOfGrid, &numOfTimeStamp);
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
    MPI_Comm_rank(MPI_COMM_WORLD, &numOfRank);
    MPI_Status status;
    int length = (int)numOfGrid;

    double recvFromLeft = 0; //接收左边的数值
    double recvFromRight = 0; //接收右边的数值

    double *current = new double[length]; // 存储当前计算得出的温度
    double *previous = new double[length]; // 存储上个时态的温度
    double *finalTemp = new double[length]; // 存储最后的结果
    for (int i = 0; i < length; i++) {
    	current[i] = 0;
    	previous[i] = 0;
    }

    int chunk = length / numOfProcess;
    int leastChunk = length % numOfProcess;

    int startingPoint = numOfRank * chunk;   //这个是每个rank的起始位置
    int endingPoint;   //这个是每个rank的终止位置
    if(numOfRank == numOfProcess - 1){
        endingPoint = length-1;
    } else {
        endingPoint = (numOfRank + 1) * chunk - 1;
    }
    int subLength = endingPoint - startingPoint + 1;//这个是每个rank的长度

    for (int t = 0; t < numOfTimeStamp; t++){
	    if (numOfRank == 0) {

	    	// 如果仅有一个process则不向右发送，并将recvFromRight设为endTemp
	    	if (numOfRank != numOfProcess - 1) {
	    		MPI_Send(&previous[endingPoint], 1, MPI_DOUBLE, numOfRank+1, t, MPI_COMM_WORLD);//把rank0最后一位发给右边
	        	MPI_Recv(&recvFromRight, 1, MPI_DOUBLE, numOfRank+1, t, MPI_COMM_WORLD, &status);//rank0接收从右边发送过来的数值
	    	} else {
	    		recvFromRight = endTemp;
	    	}
	    	
	    	// cout << "rank0 recv from right " << recvFromRight << endl;

	    	// Calculate
	    	for (int i = startingPoint; i <= endingPoint; i++) {
	        	double _front, _back;
	        	if (i == startingPoint) {
	        		_front = beginTemp;
	        	} else {
	        		_front = previous[i - 1];
	        	}
	        	if (i == endingPoint) {
	        		_back = recvFromRight;
	        	} else {
	        		_back = previous[i + 1];
	        	}
	        	
	        	current[i] = 0.5 * previous[i] + 0.25 * _front + 0.25 * _back;
	        }

	        // update previous as current
	        for (int i = startingPoint; i <= endingPoint; i++) {
	        	previous[i] = current[i];
	        }

	  //       cout << "Rank" << numOfRank << " cal: ";
			// for (int i = startingPoint; i <= endingPoint; i++) {
			// 	cout << previous[i] << ", ";
			// }
			// cout << "..." << endl;
			
			if (t == numOfTimeStamp - 1) {  // 仅当最后一个timeStep接收所有rank计算的数据并输出最终结果
				copy(current, current + subLength, finalTemp);

				for(int i = 1; i < numOfProcess; i++){ //用来接收其他rank计算好的数组
	                if(i == numOfProcess - 1){
	                    MPI_Recv(&finalTemp[i*chunk], chunk+leastChunk, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
	                }
	                else{
	                    MPI_Recv(&finalTemp[i*chunk], chunk, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
	                }
	            }

	            // print out the results
	            for (int i = 0; i < numOfGrid; i++) {
					if (i == numOfGrid - 1) {
						cout << finalTemp[i];
					} else {
						cout << finalTemp[i] << ", ";
					}
				}

				// Generate the heat1Doutput.csv output file.
				ofstream outFile;
				outFile.open("heat1Doutput.csv", ios::out);
				for (int i = 0; i < gridPoints; i++) {
					if (i == gridPoints - 1) {
						outFile << output[i];
					} else {
						outFile << output[i] << ", ";
					}
				}
			}

	    } else if (numOfRank == numOfProcess - 1) {
	    	MPI_Send(&previous[startingPoint], 1, MPI_DOUBLE, numOfRank-1, t, MPI_COMM_WORLD);//将最后一个rank的第一位发向左边
	        MPI_Recv(&recvFromLeft, 1, MPI_DOUBLE, numOfRank-1, t, MPI_COMM_WORLD, &status);//接收从左边发送过来的数值
	        
	        // cout << "rank" << numOfRank << " recv from left " << recvFromLeft << endl;

	        // Calculate
	        for (int i = startingPoint; i <= endingPoint; i++) {
	        	double _front, _back;
	        	if (i == startingPoint) {
	        		_front = recvFromLeft;
	        	} else {
	        		_front = previous[i - 1];
	        	}
	        	if (i == endingPoint) {
	        		_back = endTemp;
	        	} else {
	        		_back = previous[i + 1];
	        	}
	        	
	        	current[i] = 0.5 * previous[i] + 0.25 * _front + 0.25 * _back;
	        }


	        // update previous as current
	        for (int i = startingPoint; i <= endingPoint; i++) {
	        	previous[i] = current[i];
	        }

	  //       cout << "Rank" << numOfRank << " cal: ";
			// for (int i = startingPoint; i <= endingPoint; i++) {
			// 	cout << previous[i] << ", ";
			// }
			// cout << "..." << endl;

			if (t == numOfTimeStamp - 1) {
				MPI_Send(&current[startingPoint], subLength, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}

	    } else {
	        MPI_Send(&previous[startingPoint], 1, MPI_DOUBLE, numOfRank-1, t, MPI_COMM_WORLD);//将rank开头的数值发给左边
	        MPI_Recv(&recvFromLeft, 1, MPI_DOUBLE, numOfRank-1, t, MPI_COMM_WORLD, &status);//接收从左边发送过来的数值

	        MPI_Send(&previous[endingPoint], 1, MPI_DOUBLE, numOfRank+1, t, MPI_COMM_WORLD);//将rank最后一位发送右边
	        MPI_Recv(&recvFromRight, 1, MPI_DOUBLE, numOfRank+1, t, MPI_COMM_WORLD, &status);

	        // cout << "rank" << numOfRank << " recv from left " << recvFromLeft << ", recv from right " << recvFromRight << endl;

	        // Calculate
	        for (int i = startingPoint; i <= endingPoint; i++) {
	        	double _front, _back;
	        	if (i == startingPoint) {
	        		_front = recvFromLeft;
	        	} else {
	        		_front = previous[i - 1];
	        	}
	        	if (i == endingPoint) {
	        		_back = recvFromRight;
	        	} else {
	        		_back = previous[i + 1];
	        	}

	        	current[i] = 0.5 * previous[i] + 0.25 * _front + 0.25 * _back;
	        }

	        // update the previous as current
	        for (int i = startingPoint; i <= endingPoint; i++) {
	        	previous[i] = current[i];
	        }

	  //       cout << "Rank" << numOfRank << " cal: ";
			// for (int i = startingPoint; i <= endingPoint; i++) {
			// 	cout << previous[i] << ", ";
			// }
			// cout << "..." << endl;

			if (t == numOfTimeStamp - 1) {    // 当最后一个timestep时发送计算结果到rank0
				MPI_Send(&current[startingPoint], subLength, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}

	    }
    }

	MPI_Finalize();
    return 0;
}