/*************************************************************************
/* ECE 277: GPU Programmming 2021 Winter
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/


#define COLS 	4
#define ROWS 	4

#define RIGHT 	0
#define DOWN 	1
#define LEFT	2
#define UP 		3


short *d_action;
int size = sizeof(int);


__global__ void cuda_init() {}

__global__ void cuda_agent(int2 *cstate, short *d_action) {

	int idx = 0;
	int pos_x = cstate[idx].x, pos_y = cstate[idx].y;
	short action; 

	if (pos_y == 0) { 
		action = pos_x < COLS - 1 ? RIGHT : DOWN;
	} 
	if (pos_x == COLS - 1) { 
		action = pos_y < ROWS - 2 ? DOWN : LEFT;
	} 

	d_action[idx] = action;
}


void agent_init() {
	// allocate a short-type global memory, d_action ptr (allocated GPU)
	cudaMalloc((void **)&d_action, size);
	cuda_init <<<1, 1>>> ();
}


short* agent_action(int2* cstate) {
	// invokes an CUDA kernel (cuda_agent), cstate ptr (allocated GPU)
	cuda_agent <<<1, 1>>> (cstate, d_action);
	return d_action;
}

// cudaMemcpy(&d_action, source, size, cudaMemcpyDeviceToHost);

