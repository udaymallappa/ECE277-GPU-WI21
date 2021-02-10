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


short *d_action;

#define COLS_OF_TABLE	4
#define ROWS_OF_TABLE	4

#define NUM_OF_STATES	16
#define NUM_OF_ACTIONS	4

float *d_qtable;
// curandState *states;

/*  Host: agent_init()  
	Device: agnet_init(), qtable_init(), randstate_init();
 */ 

__global__ void Agent_init() {
	// works for multiple agents 
	__device__ float *gamma = 0.9;
	__device__ float *alpha = 0.1;
	__device__ float *epsilon = 0.001;
	
	// zero is active, nonzero is inactive (dead)
	// int agent_id = threadIdx.y + blockIdx.y * blockDim.y;
	// bool agent_status[agent_id] = 0; 
}

__global__ void Qtable_init(float *d_qtable) {

	int nx = NUM_OF_STATES * NUM_OF_ACTIONS;

	int tid_x = threadIdx.x + blockIdx.x * blockDim.x; // states for each agent
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y; // agent id 
	int tid = tid_y * nx + tid_x;

	// init Q-table Q(s, a) = 0, s in S, a in A(s)
	d_qtable[tid] = 0;

}

// __global__ void Randstate_init(currandState *d_rand_state) {
// 	// 
// 	int nx = NUM_OF_STATES * NUM_OF_ACTIONS;

// 	int tid_x = threadIdx.x + blockIdx.x * blockDim.x; // states for each agent
// 	int tid_y = threadIdx.y + blockIdx.y * blockDim.y; // agent id 
// 	int tid = tid_y * nx + tid_x;

// 	// unsigned long long seed 1234
// 	curand_init(1234, tid, 0, &d_rand_state[tid]); 

// }


void agent_init()
{
	int qtableSize = NUM_OF_ACTIONS * NUM_OF_STATES;
	// int randStateSize =  NUM_OF_STATES * NUM_OF_AGENTS;

	cudaMalloc((void **)&d_action, sizeof(short));
	cudaMalloc((void **)&d_qtable, sizeof(float) * qtableSize); // 4 acts * 16 state * 1 agent
	// cudaMalloc((void **)&states, sizeof(curandState) * randStateSize); // TODO: curandState size is float or 4*float???
	// cudaMalloc((void **)&agent_status, sizeof(bool) * NUM_OF_AGENTS);

	// dim3 grid(NUM_OF_STATES, NUM_OF_AGENTS, 1); // 
	// dim2 block(NUM_OF_ACTIONS, 1, 1);
	
	Agent_init <<<1, 1>>> ();
	Qtable_init <<<NUM_OF_STATES, NUM_OF_ACTIONS>>> (&d_qtable); // 16 states x 4 actions
	// Randstate_init <<<1, NUM_OF_STATES>>> (&state);

}

///////////////////////////////    agent_ation()    ////////////////////////////// 

// __global__ void kernel_fun(curandState *d_randstate) {
// 	int tid = blockIdx.x * blockDim.x + threadIdx.x;
// 	curand_uniform(&d_randstate[tid]);
// }

__global__ void Agent_action(int2 *cstate, short *d_action) {

	// int nx = NUM_OF_STATES * NUM_OF_ACTIONS;
	// int cstate_id = blockIdx.x + blockIdx.y * gridDim.x;
	int cstate_id = ;
	int x = cstate[cstate_id].x;
	int y = cstate[cstate_id].y;

	// the x, y coodinate will be block.x block.y in qtable
	// for each pos, there are still 4 action types;
	int tid = y * nx + x;
	int action;
	float max_qval;

	// exploration
	if (curand_uniform(1) < epsilon) {
		action = (int*) curand_uniform(1) * NUM_OF_ACTIONS;
	} else { // exploitation (greedy policy)
		action = 0;
		for (int i = 0; i < 4; ++i) {
			if (d_qtable[tid + i] > max_qval) {
				max_qval = d_qtable[tid + i];
				action = i;
			}
		}
	}
	// decide action
	d_action[cid] = action;
}


short* agent_action(int2* cstate) {

	dim3 grid(NUM_OF_STATES, NUM_OF_AGENTS, 1); 
	dim2 block(NUM_OF_ACTIONS, 1, 1);

	Agent_action <<<grid, block>>> (cstate, d_action);

	return d_action;
}

/////////////////////////////    agent_update()    ///////////////////////////////

__global__ void Agent_update(int2* cstate, int2* nstate, float *rewards) {
	// observe next state S' and R
	// Q(S, A) <- Q(S, A) + alpha[R + gamma * max Q(S', a) - Q(S, A)]

	int cid =;
	int x0 = cstate[cid].x;
	int y0 = cstate[cid].y;

	int c_tid = y0 * nx + x0;

	int nid = ;
	int x1 = nstate[nid].x;
	int y1 = nstate[nid].y;

	int n_tid = y1 * nx + x1;
	float max_qval;

	for (int i = 0; i < NUM_OF_ACTIONS; ++i) {
		if (d_qtable[n_tid + i] > max_qval) {
			max_qval = d_qtable[n_tid + i];		
		}
	}
	// S <- S'
	d_qtable[c_tid + d_action] += alpha * (rewards + gamma * max_qval - d_qtable[c_tid + d_action]);

	// update state
	if (rewards[idx] == 0) { // 
		cstate[idx] = nstate[id];
	} else {
		cstate[idx].x = 0;
		cstate[idx].y = 0;
	}
}


void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	// add your codes
	Agent_update <<<1, 1>>> (cstate, nstate, rewards);
}


/////////////////////////////    adjust_epsilon()    ///////////////////////////////



float agent_adjustepsilon()
{
	// add your codes
	epsion -= 0.001;
	return epsilon;
}

