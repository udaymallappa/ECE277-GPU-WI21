/*************************************************************************
/* ECE 277: GPU Programmming 2021 WINTER quarter
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/* 
/* Midterm
/* Student: Yifan Wang (A53298382)
/* Email: yiw021@ucsd.edu
/*************************************************************************/

#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>


#define ACTIONS		4
#define NUM_AGENTS	512

#define COLS		46
#define ROWS		46
#define QSIZE 		COLS * ROWS * ACTIONS

#define THREADS		256

#define GAMMA		0.9
#define ALPHA		0.5
#define	EPSILON		1.0
#define DELTA_EPS	0.01

#define EPS_CEIL	1.0
#define EPS_BOTTOM  0.0


short *d_action;
curandState *d_states;

bool  *d_active;
float *d_qtable;

float epsilon;


//////////////////////////	agent_init()  //////////////////////////

// <<<NUM_AGENTS * ACTIONS / THREADS, THREADS >>>
__global__ void Init_agent(curandState *d_states, bool *d_active)
{
	unsigned int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (agent_id < NUM_AGENTS) {
		curand_init(clock() + agent_id, agent_id, 0, &d_states[agent_id]);
		d_active[agent_id] = 1;
	}
}


// occupency
__global__ void Init_qtable(float *d_qtable)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int nx = gridDim.x * blockDim.x;
	unsigned int tid = ix + iy * nx; 

	// make sure it won't be out of range size
	if (tid < QSIZE) {
		d_qtable[tid] = 0.0f;
	}
}

void agent_init()
{
	// clear action + initQ table + self initialization
	epsilon = EPSILON;

	cudaMalloc((void **)&d_action, NUM_AGENTS * sizeof(short));
	
	cudaMalloc((void **)&d_states, NUM_AGENTS * sizeof(curandState));
	cudaMalloc((void **)&d_active, NUM_AGENTS * sizeof(bool));

	dim3 block(THREADS, 1, 1);
	dim3 grid(NUM_AGENTS / block.x, 1, 1);
	Init_agent << <grid, block >> > (d_states, d_active);

	cudaMalloc((void **)&d_qtable, QSIZE * sizeof(float));

	dim3 qblock(THREADS, 1, 1);
	dim3 qgrid(QSIZE / block.x + 1); // COLS 46 * ROWS 46 * ACTIONS 4 
	Init_qtable <<< qgrid , qblock >>> (d_qtable);	

}


//////////////////////////	agent_init_episode() //////////////////////////

// <<<NUM_AGENTS * ACTIONS / THREADS, THREADS >>>
__global__ void Init_epsiode(bool *d_active) {
	// agent 1 alive, 0 dead;
	unsigned int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
	d_active[agent_id] = 1;
}


void agent_init_episode() {
	// set all agents in active status
	dim3 block(THREADS, 1, 1);
	dim3 grid(NUM_AGENTS / block.x, 1, 1);
	Init_epsiode <<<grid, block >>> (d_active);
}


//////////////////////////	adjust_epsilon() //////////////////////////


float agent_adjustepsilon()
{
	if (epsilon > EPS_CEIL) {
		epsilon = EPS_CEIL;
	}
	else if (epsilon < EPS_BOTTOM) {
		epsilon = EPS_BOTTOM;
	}
	else {
		epsilon -= (float)DELTA_EPS;
	}
	return epsilon;
}


//////////////////////////	agent_action() //////////////////////////


// <<<NUM_AGENTS * ACTIONS / THREADS, THREADS >>>
__global__ void Agent_action(int2 *cstate, short *d_action, curandState *d_states, float epsilon, float *d_qtable, bool *d_active) {
    
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int nx = gridDim.x * blockDim.x;

	int tid = iy * nx + ix;
	int agent_id = tid / ACTIONS;

	float rand_state = curand_uniform(&d_states[agent_id]);
	if (rand_state < epsilon) {
		short action = (short)(curand_uniform(&d_states[agent_id]) * ACTIONS);
		if (action == 4) action = 0; // (0, 1], for keeping uniform distributed to make action==4 to 0
		d_action[agent_id] = action;
	}

	else {
		// memory shared
		__shared__ float qval_cache[THREADS];
		__shared__ short action_cache[THREADS];
	
		int sid = threadIdx.x;
		int aid = sid % ACTIONS;
		action_cache[sid] = aid;
	
		int x = cstate[agent_id].x, y = cstate[agent_id].y;
		int qid = (y * COLS + x) * ACTIONS;
		qval_cache[sid] = d_qtable[qid + aid];
	
		__syncthreads();
	
		unsigned int stride = ACTIONS / 2;
	
		// reduction, best action
		#pragma unroll
		while (stride != 0) {
			if (aid < stride) {
				if (qval_cache[sid] < qval_cache[sid + stride])  {
					qval_cache[sid] = qval_cache[sid + stride];
					action_cache[sid] = action_cache[sid + stride];
				}
			} 
			__syncthreads();
			stride /= 2;
		} 
	
		if (sid % ACTIONS == 0) {
			d_action[agent_id] = action_cache[sid];
		}
	
	}
}

short* agent_action(int2* cstate) {
	// do exploration or exploitation
	dim3 block(THREADS, 1, 1);
	dim3 grid(NUM_AGENTS * ACTIONS / block.x, 1, 1);
	Agent_action <<< grid, block >>> (cstate, d_action, d_states, epsilon, d_qtable, d_active); 
	return d_action;
}



//////////////////////////	agent_update() //////////////////////////

// <<<NUM_AGENTS * ACTIONS / THREADS, THREADS >>>
__global__ void Agent_update(int2* cstate, int2* nstate, float *rewards, float *d_qtable, short *d_action, bool *d_active)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int nx = gridDim.x * blockDim.x;
	int tid = iy * nx + ix;
	int agent_id = tid / ACTIONS;

	if (d_active[agent_id] == 1) {
		
		unsigned int x0 = cstate[agent_id].x, y0 = cstate[agent_id].y;
		unsigned int c_qid = (y0 * COLS + x0) * ACTIONS + (int)d_action[agent_id];

		float gamma_item = 0;

		if (rewards[agent_id] != 0) {
			d_qtable[c_qid] += ALPHA * (rewards[agent_id] + gamma_item - d_qtable[c_qid]);
		}

		else {
			// memory shared 
			__shared__ float qval_cache[THREADS];
		
			int sid = threadIdx.x;
			int aid = sid % ACTIONS;
		
			// next state
			int x = nstate[agent_id].x, y = nstate[agent_id].y;
			int qid = (y * COLS + x) * ACTIONS;
			qval_cache[sid] = d_qtable[qid + aid];
		
			// reduction, max qval
			unsigned int stride = ACTIONS / 2;
		
			#pragma unroll
			while (stride != 0) {
				if (aid < stride) {
					if (qval_cache[sid] < qval_cache[sid + stride])  {
						qval_cache[sid] = qval_cache[sid + stride];
					}
				} 
				__syncthreads();
				stride /= 2;
			} 

			if (sid % ACTIONS == 0)  {
				d_qtable[c_qid] += ALPHA * (rewards[agent_id] +  GAMMA * qval_cache[sid] - d_qtable[c_qid]);
			}
		}
	}
}

void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	dim3 block(THREADS, 1, 1);
	dim3 grid(NUM_AGENTS * ACTIONS / block.x, 1, 1);
	Agent_update <<< grid, block >>> (cstate, nstate, rewards, d_qtable, d_action, d_active);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////


/** CUDA Dynamic Parallelism
* @brief it will be called in __global__ Agent_action and Agent_update
*	for agent_id to calculate (.x) greedy_action and (.y) max_qval
* @param float2 d_actval 
* @return __device__ void
*/

// __inline__ __device__ void Get_qAction_qMaxVal(int2 *state, float *d_qtable, float2 *d_actval, unsigned int agent_id)
// {
// 	// exploitation (greedy policy)

// 	// located position on q_table
// 	unsigned int x = state[agent_id].x;
// 	unsigned int y = state[agent_id].y;

// 	// memory shared
// 	__shared__ float qval_cache[ACTIONS]; // 4 actions  
// 	__shared__ short action_cache[ACTIONS];

// 	unsigned int aid = threadIdx.x; // action_id
// 	action_cache[aid] = (short)threadIdx.x;

// 	unsigned int q_id = (y * COLS + x) * ACTIONS;
// 	qval_cache[aid] = d_qtable[q_id + aid];

// 	__syncthreads();

// 	// reduction for getting the max val and action
// 	unsigned int stride = blockDim.x / 2; // 4 actions / 2

// #pragma unroll
// 	while (stride != 0) {
// 		if (aid < stride && qval_cache[aid] < qval_cache[aid + stride]) {
// 			// keep larger values in left cache
// 			qval_cache[aid] = qval_cache[aid + stride];
// 			action_cache[aid] = action_cache[aid + stride];
// 		}
// 		__syncthreads();
// 		stride /= 2;
// 	}
// 	// update: .x action; .y max_qval.
// 	d_actval[agent_id].x = action_cache[0];
// 	d_actval[agent_id].y = qval_cache[0];
// }


////////////////////////////////////////////////////////////////////////////////////////////////////////
