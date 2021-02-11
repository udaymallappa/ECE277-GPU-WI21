/*************************************************************************
/* ECE 277: GPU Programmming 2021 WINTER
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/

#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>


#define COLS	4
#define ROWS	4

#define NUM_OF_AGENTS	1
#define NUM_OF_ACTIONS	4

#define GAMMA		0.9
#define ALPHA		0.1
#define	EPSILON		1.0
#define DELTA_EPS	0.001

short *d_action;
curandState *d_randstate;
float *d_qtable;

float epsilon;
float *d_epsilon;


///////////////////////////////    agent_init()    ////////////////////////////// 

__global__ void Agent_init(float *d_epsilon) {
	// works for init agents and initalize parameters
	*d_epsilon = 1.000f;
}

__global__ void Qtable_init(float *d_qtable) {
	// it's 4 x board_size, as the qtable includes 4 actions at each position.
	// init Q-table Q(s, a) = 0, s in S, a in A(s)
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	d_qtable[tid] = 0;
}

__global__ void Randstate_init(curandState *d_randstate) {
	// all the threads need to maintain their own states 
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(clock() + tid, tid, 0, &d_randstate[tid]);
}

void agent_init()
{
	int qSize = NUM_OF_ACTIONS * COLS * ROWS;
	int randSize = NUM_OF_AGENTS;

	cudaMalloc((void **)&d_qtable, sizeof(float) * qSize);
	cudaMalloc((void **)&d_randstate, sizeof(curandState) * randSize);

	cudaMalloc((void **)&d_action, sizeof(short) * NUM_OF_AGENTS);
	cudaMalloc((void **)&d_epsilon, sizeof(float));

	Agent_init << <1, NUM_OF_AGENTS >> > (d_epsilon);
	Qtable_init << <COLS * ROWS, NUM_OF_ACTIONS >> > (d_qtable);
	Randstate_init << <1, NUM_OF_AGENTS >> > (d_randstate);

}

///////////////////////////////    agent_ation()    ////////////////////////////// 

// __global__ void kernel_fun(curandState *d_randstate) {
// 	int tid = blockIdx.x * blockDim.x + threadIdx.x;
// 	curand_uniform(&d_randstate[tid]);
// }

__global__ void Agent_action(int2 *cstate, short *d_action, curandState *d_randstate, float *d_epsilon, float *d_qtable) {

	int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
	int x = cstate[agent_id].x;
	int y = cstate[agent_id].y;

	// the x, y coodinate will be block.x block.y in qtable
	// for each pos, there are still 4 action types;
	short action;

	// exploration
	float rand_state = curand_uniform(&d_randstate[agent_id]);

	if (rand_state < *d_epsilon) {
		// float div = 1.000f / ((float)NUM_OF_ACTIONS);
		// action = (int)(rand_state / div);
		action = (short)(curand_uniform(&d_randstate[agent_id]) * (float)NUM_OF_ACTIONS);
	}
	else {
		// exploitation (greedy policy)
		float max_qval;
		int qid = (y * COLS + x) * NUM_OF_ACTIONS;
		for (int i = 0; i < NUM_OF_ACTIONS; ++i) {
			if (d_qtable[qid + i] > max_qval) {
				max_qval = d_qtable[qid + i];
				action = (short)i;
			}
		}
	}

	// decide the action
	*d_action = action;
}


short* agent_action(int2* cstate) {
	// do exploration or exploitation
	Agent_action << <1, NUM_OF_AGENTS >> > (cstate, d_action, d_randstate, d_epsilon, d_qtable);
	return d_action;
}

/////////////////////////////    agent_update()    ///////////////////////////////

__global__ void Agent_update(int2* cstate, int2* nstate, float *rewards, float *d_qtable, short *d_action)
{
	// observe next state S' and R
	int agent_id = blockIdx.x * blockDim.x + threadIdx.x;

	int x0 = cstate[agent_id].x;
	int y0 = cstate[agent_id].y;

	int x1 = nstate[agent_id].x;
	int y1 = nstate[agent_id].y;

	// next state (n+1)
	int n_qid = (y1 * COLS + x1) * NUM_OF_ACTIONS;
	float max_qval = d_qtable[n_qid];
	// i start from 1 as the i = 0 has been assign as init max_qval
	for (int i = 1; i < NUM_OF_ACTIONS; ++i) {
		if (d_qtable[n_qid + i] > max_qval) {
			max_qval = d_qtable[n_qid + i];
		}
	}

	// update q_table of current state (n) by max val of next state
	// Q(S, A) <- Q(S, A) + alpha[R + gamma * max Q(S', a) - Q(S, A)]
	int c_qid = (y0 * COLS * x0) + *d_action;
	d_qtable[c_qid] += ALPHA * (rewards[agent_id] + GAMMA * max_qval - d_qtable[c_qid]);

	// update state to next
	if (rewards[agent_id] == 0) {
		// agent status: active
		cstate[agent_id] = nstate[agent_id];
	}
	else {
		// agent status: inactive 
		cstate[agent_id].x = 0;
		cstate[agent_id].y = 0;
	}
}


void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	// add your codes
	Agent_update << <1, NUM_OF_AGENTS >> > (cstate, nstate, rewards, d_qtable, d_action);
}


/////////////////////////////    adjust_epsilon()    ///////////////////////////////

__global__ void Adjust_epsilon(float *d_epsilon) {
	if (*d_epsilon > 1.000f) {
		*d_epsilon = 1.000f;
	}
	else if (*d_epsilon < 0.100f) {
		*d_epsilon = 0.100f;
	}
	else {
		*d_epsilon -= DELTA_EPS;
	}
}


float agent_adjustepsilon()
{
	Adjust_epsilon << <1, 1 >> > (d_epsilon);
	cudaMemcpy(&epsilon, d_epsilon, sizeof(float), cudaMemcpyDeviceToHost);
	return epsilon;
}
