
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include<fstream>
#include<iomanip>
#include <windows.h>
using namespace std;
//location_x, location_y, location_z, width, height, depth, fixed temperature
typedef struct FixT {

	dim3 loc;
	dim3 size;
	float t;
};
//config file
typedef struct Conf {
	float k;
	int times;
	dim3 size;
	float defaultT;
	vector<FixT*> fixts;
};

struct Conf conf;
int conIndex = 0;

/*
split string
*/
vector<string> split(string str, string pattern) {
	vector<string> result;
	int pos;
	str += pattern;
	for (int i = 0; i < str.size(); i++)
	{
		pos = str.find(pattern, i);
		if (pos<str.size())
		{
			result.push_back(str.substr(i, pos - i));
			i = pos + pattern.size() - 1;
		}
	}
	return result;
}

//get fixed temperature struct data
FixT* getFixT(vector<string> &vec) {
	FixT *fixt = (FixT*)malloc(sizeof(FixT));
	int index = 0;
	fixt->loc.x = atoi(vec[index++].c_str());
	fixt->loc.y = atoi(vec[index++].c_str());
	fixt->loc.z = 0;
	if (vec.size()>5)
	{
		fixt->loc.z = atoi(vec[index++].c_str());
	}
	fixt->size.x = atoi(vec[index++].c_str());
	fixt->size.y = atoi(vec[index++].c_str());
	fixt->size.z = 1;
	if (vec.size()>5)
	{
		fixt->size.z = atoi(vec[index++].c_str());
	}
	fixt->t = atof(vec[index++].c_str());
	return fixt;
}
//read data from file
void readData(char *file) {
	char line[1024];
	ifstream readFile(file);
	int rown = 0;
	while (!readFile.eof())
	{
		readFile.getline(line, 1024);
		string str(line);

		if (line[0] != '#')
		{
			string str(line);
			if (str=="2D"||str=="3D")
			{
				continue;
			}
			vector<string> vec = split(str, ",");
			if (vec.size()==1)
			{
				if (rown>3)
				{
					rown = 0;
					conIndex++;
				}
			}
			switch (rown)
			{
			case 0:
				conf.k = atof(vec[0].c_str());
				if (conf.k>1)
				{
					conf.k /= 100;
				}
				break;
			case 1:
				conf.times= atoi(vec[0].c_str());
				break;
			case 2:
				conf.size.x = atoi(vec[0].c_str());
				conf.size.y = atoi(vec[1].c_str());
				conf.size.z = 1;
				if (vec.size()==3)
				{
					conf.size.z = atoi(vec[2].c_str());
				}
				break;
			case 3:
				conf.defaultT = atof(vec[0].c_str());
				break;
			default:
				FixT*fix = getFixT(vec);
				conf.fixts.push_back(fix);
				break;
			}
			rown++;
			

		}

	}
	readFile.close();
}


//out data to file
void outData(float **data,Conf*conf,string file) {
	FILE* fp = fopen(file.c_str(), "w+");
	for (int z = 0; z < conf->size.z; z++)
	{
		for (int y = 0; y < conf->size.y; y++)
		{
			for (int x = 0; x < conf->size.x; x++)
			{
				float d = data[z][x + y*conf->size.x];
				fprintf(fp, "%.4f", d);
				if (x<conf->size.x-1)
				{
					fprintf(fp,",");
				}
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}


//gpu compute
__global__ void simulat(float**devr,int x,int y,int z,float k) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idz = threadIdx.z+ blockDim.z*blockIdx.z;
	
	if (idx>=x||idy>=y||idz>=z)
	{
		return;
	}
	int dirs[][3] = {
		{1,0,0},
		{ -1,0,0 },
		{ 0,1,0 },
		{ 0,-1,0 },
		{ 0,0,1 },
		{ 0,0,-1 },
	};
	
	

	float old=devr[idz][idx + idy*x];
	if (old<0)
	{
		return;
	}
	float sum = 0;
	
	for (int i = 0; i < 6; i++)
	{
		int dx = dirs[i][0] + idx;
		int dy = dirs[i][1] + idy;
		int dz = dirs[i][2] + idz;
		if (dx>=0&&dx<x&&dy>=0&&dy<y&&dz>=0&&dz<z)
		{
			sum += abs(devr[dz][dx + dy*x]) - old;
		}
	}
	if (old>0)
	{
		devr[idz][idx + idy*x] = old + k*sum;
	}


	
	
	

}
//simulate starting
void heat(Conf*conf,string file) {

	float **room;
	float **devr;
	

	int x = conf->size.x;
	int y = conf->size.y;
	int z = conf->size.z;
	room = (float**)malloc(sizeof(float*)*z);//cpu mem
	for (int i = 0; i < z; i++)
	{
		do {
			room[i] = (float*)malloc(sizeof(float)*x*y);
			
		} while (room[i]==0);
		
		
	}

	for (int i = 0; i < z; i++)
	{
		for (int j = 0; j < x*y; j++)
		{
			room[i][j] = conf->defaultT;

		}


	}
	//init data
	for (int i = 0; i < conf->fixts.size(); i++)
	{
		FixT*fix = conf->fixts[i];
		for (int z1 = fix->loc.z; z1 < fix->size.z + fix->loc.z; z1++)
		{
			for (int y1 = fix->loc.y; y1 < fix->size.y + fix->loc.y; y1++)
			{
				for (int x1 = fix->loc.x; x1 < fix->size.x + fix->loc.x; x1++)
				{
					int index = x1 + y1*x;
					room[z1][index] = -fix->t;
				}
			}
		}
	}
	//gpu array
	float **tem = (float**)malloc(sizeof(float*)*z);
	for (int i = 0; i < z; i++)
	{
		cudaMalloc((void**)&tem[i], sizeof(float)*x*y);
		cudaMemcpy(tem[i], room[i], sizeof(float) * x*y, cudaMemcpyHostToDevice);

	}

	cudaMalloc((void**)&devr, sizeof(float*)*z);
	cudaMemcpy(devr, tem, sizeof(float*) * z, cudaMemcpyHostToDevice);
	
	//gpu thread init
	int sx = 5;
	int sy = 5;
	int sz = 5;
	dim3 blocks(x/sx+(x%sx>0),y/sy+(y%sy>0) , z / sz + (z%sz>0));
	dim3 threads(sx,sy,sz);
	
	for (int i = 0; i < conf->times; i++)
	{
	
		simulat << <blocks, threads >> >(devr, x, y, z, conf->k);
		
		int cudaStatus = cudaThreadSynchronize();
		if (cudaStatus != cudaSuccess) {
			cout << "error" << endl;
		}
	}
	
		
	

	for (int i = 0; i < z; i++)
	{
		
		cudaMemcpy( room[i], tem[i], sizeof(float) * x*y, cudaMemcpyDeviceToHost);
		for (int j = 0; j < x*y; j++)
		{
			if (room[i][j]<0)
			{
				room[i][j] = -room[i][j];
			}
		}
		cudaFree(tem[i]);
	}
	cudaFree(devr);

	outData(room,conf,file);
	for (int i = 0; i < z; i++)
	{
		free(room[i]);
	}
	free(room);

}

int main(int argc,char **argv)
{
  
	readData(argv[1]);
	heat(&conf, "heatOutput.csv");
	cout << "The End!" << endl;
	getchar();
    return 0;
}

