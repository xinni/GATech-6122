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
    double *initialTemp = new double[length];
    double *initialTempBuf = new double[length];
    double *sendingBuffer = new double[length];
    double *computedTemp = new double[length];
    double recvFromLeft; //接收左边的数值
    double recvFromRight; //接收右边的数值

    double *replaceTemp = new double[length];
    for(int i = 0;i < length;i++){
        initialTemp[i] = 0;//这个是最初始的数组
    }

    for(int j = 0;j < length;j++){
        initialTempBuf[j] = 0;//这个是用来计算的数组
    }

    for(int x = 0;x < length;x++){
        replaceTemp[x] = 0;//这个是用来代替初始数组的数组
    }
    int chunk = length / numOfProcess;
    int leastChunk = length % numOfProcess;

    int startingPoint = numOfRank*chunk;//这个是每个rank的起始位置
    int endingPoint;//这个是每个rank的终止位置
    if(numOfRank == numOfProcess - 1){
        endingPoint = length-1;
    }
    else{
        endingPoint = (numOfRank+1)*chunk-1;
    }
    int subLength = endingPoint-startingPoint+1;//这个是每个rank的长度
    for (int t=0;t<numOfTimeStamp;t++){
        if(numOfRank==0){
            int num;
            MPI_Send(&initialTemp[endingPoint],1,MPI_DOUBLE,numOfRank+1,0,MPI_COMM_WORLD);//把rank0最后一位发给右边
            MPI_Recv(&recvFromRight,1,MPI_DOUBLE,numOfRank+1,0,MPI_COMM_WORLD,&status);//rank0接收从右边发送过来的数值
            initialTempBuf[endingPoint+1] = recvFromRight;//把接收的数值放到用于计算的数组对应位置中
            for(num=startingPoint;num<=endingPoint;num++){
                if(num==0){
                    computedTemp[num] = 0.25*beginTemp+0.5*initialTempBuf[num]+0.25*initialTempBuf[num+1];
                }
                else if(num==length-1){
                    computedTemp[num] = 0.25*initialTempBuf[num-1]+0.5*initialTempBuf[num]+0.25*endTemp;
                }
                else{
                    computedTemp[num] = 0.25*initialTempBuf[num-1]+0.5*initialTempBuf[num]+0.25*initialTempBuf[num+1];
                }
                replaceTemp[num]= computedTemp[num];
//            std::cout<<"rank: "<<numOfRank<<std::endl;
//            std::cout<<initialTempBuf[num]<<std::endl;
            }
            for(int i=1;i<numOfProcess;i++){ //用来接收其他rank计算好的数组
                if(i == numOfProcess-1){
                    MPI_Recv(&replaceTemp[i*chunk],chunk+leastChunk,MPI_DOUBLE,i,0,MPI_COMM_WORLD,&status);
                }
                else{
                    MPI_Recv(&replaceTemp[i*chunk],chunk,MPI_DOUBLE,i,0,MPI_COMM_WORLD,&status);
                }
            }
            for(int i=0;i<length;i++){
                initialTemp[i] = replaceTemp[i];
                initialTempBuf[i] = replaceTemp[i];
            }
            for (int i=0;i<length;i++){
                std::cout<<"rank: "<<i<< " is " << initialTemp[i] << std::endl;
            }

        }
        else if(numOfRank==numOfProcess-1){
            int num;
            MPI_Send(&initialTemp[startingPoint],1,MPI_DOUBLE,numOfRank-1,0,MPI_COMM_WORLD);//将最后一个rank的第一位发向左边
            MPI_Recv(&recvFromLeft,1,MPI_DOUBLE,numOfRank-1,0,MPI_COMM_WORLD,&status);//接收从左边发送过来的数值
            initialTempBuf[startingPoint-1] = recvFromLeft;//把接收数值放入计算数组的对应位置中
            for(num=startingPoint;num<=endingPoint;num++){
                if(num==0){
                    computedTemp[num] = 0.25*beginTemp+0.5*initialTempBuf[num]+0.25*initialTempBuf[num+1];
                }
                else if(num==length-1){
                    computedTemp[num] = 0.25*initialTempBuf[num-1]+0.5*initialTempBuf[num]+0.25*endTemp;
                }
                else{
                    computedTemp[num] = 0.25*initialTempBuf[num-1]+0.5*initialTempBuf[num]+0.25*initialTempBuf[num+1];
                }
//            std::cout<<"rank: "<<numOfRank<<std::endl;
//            std::cout<<initialTempBuf[num]<<std::endl;
            }
            MPI_Send(&computedTemp[startingPoint],subLength,MPI_DOUBLE,0,0,MPI_COMM_WORLD);//将计算好的数组发送给rank0
        }
        else{
            int num;
            MPI_Send(&initialTemp[startingPoint],1,MPI_DOUBLE,numOfRank-1,0,MPI_COMM_WORLD);//将rank开头的数值发给左边
            MPI_Recv(&recvFromLeft,1,MPI_DOUBLE,numOfRank-1,0,MPI_COMM_WORLD,&status);//接收从左边发送过来的数值
            initialTempBuf[startingPoint-1] = recvFromLeft;

            MPI_Send(&initialTemp[endingPoint],1,MPI_DOUBLE,numOfRank+1,0,MPI_COMM_WORLD);//将rank最后一位发送右边
            MPI_Recv(&recvFromRight,1,MPI_DOUBLE,numOfRank+1,0,MPI_COMM_WORLD,&status);
            initialTempBuf[endingPoint+1] = recvFromRight;

            for(num=startingPoint;num<=endingPoint;num++){
                if(num==0){
                    computedTemp[num] = 0.25*beginTemp+0.5*initialTempBuf[num]+0.25*initialTempBuf[num+1];
                }
                else if(num==length-1){
                    computedTemp[num] = 0.25*initialTempBuf[num-1]+0.5*initialTempBuf[num]+0.25*endTemp;
                }
                else{
                    computedTemp[num] = 0.25*initialTempBuf[num-1]+0.5*initialTempBuf[num]+0.25*initialTempBuf[num+1];
                }
//            std::cout<<"rank: "<<numOfRank<<std::endl;
//            std::cout<<initialTempBuf[num]<<std::endl;
            }
            MPI_Send(&computedTemp[startingPoint],subLength,MPI_DOUBLE,0,0,MPI_COMM_WORLD);

        }
    }
    MPI_Finalize();
    return 0;
}