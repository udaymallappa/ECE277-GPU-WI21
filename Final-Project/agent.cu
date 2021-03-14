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

#define GAMMA		0.9f		// Discount factor for past rewards
#define ALPHA		0.5f		// Learning rate
#define LAMBDA		0.9f

#define	EPSILON		1.0f		// Epsilon greedy parameter
#define DELTA_EPS	0.01f 
#define EPS_MAX		1.0f
#define EPS_MIN		0.0f


short *d_action;
curandState *d_states;

bool  *d_active;
float *d_qtable;

float epsilon;

struct eligiTraces {
	int steps = 0;
	int x[COLS * ROWS];
	int y[COLS * ROWS];
	int hist_actions[COLS * ROWS];
	float hist_traces[COLS * ROWS];
};
eligiTraces *d_traces;

//////////////////////////	agent_init()   //////////////////////////

// <<< NUM_AGENTS * ACTIONS / THREADS, THREADS >>>
__global__ void Init_agent(curandState *d_states, bool *d_active)
{
	unsigned int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (agent_id < NUM_AGENTS) {
		curand_init(clock() + agent_id, agent_id, 0, &d_states[agent_id]);
		d_active[agent_id] = 1;
	}
}


// Occupancy 
// <<< QSIZE / THREADS + 1, THREADS >>>	QSIZE = COLS 46 * ROWS 46 * ACTIONS 4 
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
	cudaMalloc((void **)&d_traces, NUM_AGENTS * sizeof(eligiTraces)); // added for Q(lambda)

	cudaMalloc((void **)&d_states, NUM_AGENTS * sizeof(curandState));
	cudaMalloc((void **)&d_active, NUM_AGENTS * sizeof(bool));

	dim3 block(THREADS, 1, 1);
	dim3 grid(NUM_AGENTS / block.x, 1, 1);
	Init_agent << <grid, block >> > (d_states, d_active);

	cudaMalloc((void **)&d_qtable, QSIZE * sizeof(float));

	dim3 qblock(THREADS, 1, 1);
	dim3 qgrid(QSIZE / block.x + 1); // COLS 46 * ROWS 46 * ACTIONS 4 
	Init_qtable << < qgrid, qblock >> > (d_qtable);

}


//////////////////////////	agent_init_episode()	//////////////////////////

// <<< NUM_AGENTS * ACTIONS / THREADS, THREADS >>>
__global__ void Init_epsiode(bool *d_active) {
	// agent 1 alive, 0 dead;
	unsigned int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
	d_active[agent_id] = 1;
}


void agent_init_episode() {
	// set all agents in active status
	dim3 block(THREADS, 1, 1);
	dim3 grid(NUM_AGENTS / block.x, 1, 1);
	Init_epsiode << <grid, block >> > (d_active);
}


//////////////////////////	adjust_epsilon()	//////////////////////////

float agent_adjustepsilon()
{
	if (epsilon > EPS_MAX) {
		epsilon = EPS_MAX;
	}
	else if (epsilon < EPS_MIN) {
		epsilon = EPS_MIN;
	}
	else {
		epsilon -= DELTA_EPS;
	}
	return epsilon;
}


//////////////////////////	agent_action()	//////////////////////////

// <<< NUM_AGENTS * ACTIONS / THREADS, THREADS >>>
__global__ void Agent_action(int2 *cstate, short *d_action, curandState *d_states, float epsilon, float *d_qtable, bool *d_active) {

	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int nx = gridDim.x * blockDim.x;

	unsigned int tid = iy * nx + ix;
	unsigned int agent_id = tid / ACTIONS;

	float rand_state = curand_uniform(&d_states[agent_id]);
	if (rand_state < epsilon) {
		// use 1 - curand_uniform to change range from (0, 1] to (1, 0], so it won't get action==4
		d_action[agent_id] = (short)((1.0f - curand_uniform(&d_states[agent_id])) * ACTIONS);
	}

	else {
		// memory shared
		__shared__ float qval_cache[THREADS];
		__shared__ short action_cache[THREADS];

		unsigned int sid = threadIdx.x;
		unsigned int aid = sid & (ACTIONS - 1); // aid = sid % ACTIONS; // 0123 0123 ...
		action_cache[sid] = aid;

		unsigned int x = cstate[agent_id].x, y = cstate[agent_id].y;
		unsigned int qid = (y * COLS + x) * ACTIONS;
		qval_cache[sid] = d_qtable[qid + aid];

		__syncthreads();

		unsigned int stride = ACTIONS >> 1; // ACTIONS / 2;

											// reduction, best action
#pragma unroll
		while (stride != 0) {
			if (aid < stride) {
				if (qval_cache[sid] < qval_cache[sid + stride]) {
					qval_cache[sid] = qval_cache[sid + stride];
					action_cache[sid] = action_cache[sid + stride];
				}
			}
			__syncthreads();
			stride = stride >> 1; // stride /= 2;
		}

		if (aid == 0) { // if (sid &(ACTIONS - 1) == 0)
			d_action[agent_id] = action_cache[sid]; // 0___ 0___ ...
		}

	}
}


short* agent_action(int2* cstate) {
	// do exploration or exploitation
	dim3 block(THREADS, 1, 1);
	dim3 grid(NUM_AGENTS * ACTIONS / block.x, 1, 1);
	Agent_action << < grid, block >> > (cstate, d_action, d_states, epsilon, d_qtable, d_active);
	return d_action;
}


//////////////////////////	agent_update()	//////////////////////////

// __inline__ __device__ void Update_eTraces(unsigned int *agent_id, short *d_action, int2 *cstate, eligiTraces *d_traces) {
// 	int idx = d_traces[agent_id]->steps;
// 	d_traces[agent_id]->hist_actions[idx] = d_action[agent_id];
// 	d_traces[agent_id]->x[idx] = cstate[action_id].x;
// 	d_traces[agent_id]->y[idx] = cstate[action_id].y;
// 	d_traces[agent_id]->steps = idx + 1;
// }

// __inline__ __device__ void TD_lambda(float *delta, unsigned int *agent_id, eligiTraces *d_traces, float *d_qtable) {
// 	// init
// 	int n = d_traces[agent_id]->steps;
// 	d_traces[agent_id]->hist_traces[n - 1] = 1;
// 	// backward trace
// 	for (int i = n - 1; i > 0; --i) {
// 		int x = d_traces[agent_id]->x[i];
// 		int y = d_traces[agent_id]->y[i];
// 		int qid = (y * COLS + x) * ACTIONS + (int)d_traces[agent_id]->hist_actions[i];
// 		d_qtable[qid] += ALPHA * delta * d_traces[agent_id]->hist_traces[i];
// 		d_traces[agent_id]->hist_traces[i - 1] = GAMMA * LAMBDA * d_traces[agent_id]->hist_traces[i];
// 	}
// }

// <<< NUM_AGENTS * ACTIONS / THREADS, THREADS >>>
__global__ void Agent_update(int2* cstate, int2* nstate, float *rewards, float *d_qtable, short *d_action, bool *d_active, eligiTraces *d_traces)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int nx = gridDim.x * blockDim.x;
	unsigned int tid = iy * nx + ix;
	unsigned int agent_id = tid / ACTIONS;

	// TD(·)
	if (d_active[agent_id] == 1) {

		unsigned int x0 = cstate[agent_id].x, y0 = cstate[agent_id].y;
		unsigned int cur_qid = (y0 * COLS + x0) * ACTIONS + (int)d_action[agent_id];
		unsigned int x1 = nstate[agent_id].x, y1 = nstate[agent_id].y;
		unsigned int nxt_qid = (y1 * COLS + x1) * ACTIONS;

		// update traces 
		// Update_eTraces(agent_id, d_action, cstate, d_traces);
		int idx = (int)d_traces[agent_id].steps;
		d_traces[agent_id].hist_actions[idx] = d_action[agent_id];
		d_traces[agent_id].x[idx] = cstate[agent_id].x;
		d_traces[agent_id].y[idx] = cstate[agent_id].y;
		d_traces[agent_id].steps = idx + 1;

		// catch mine
		if (rewards[agent_id] == -1) {
			d_qtable[cur_qid] += ALPHA * (rewards[agent_id] - d_qtable[cur_qid]);
		}

		// catch flag
		else if (rewards[agent_id] == 1) {
			float q_max = d_qtable[nxt_qid];
			for (int i = 1; i < ACTIONS; ++i) {
				if (d_qtable[nxt_qid + i] > q_max) {
					q_max = d_qtable[nxt_qid + i];
				}
			}
			float delta = rewards[agent_id] + GAMMA * q_max - d_qtable[cur_qid];

			// TD(lambda)
			// TD_lambda(delta, agent_id, d_traces, d_qtable);
			// init
			int n = d_traces[agent_id].steps;
			d_traces[agent_id].hist_traces[n - 1] = 1;
			// backward trace
			for (int i = n - 1; i > 0; --i) {
				int x = d_traces[agent_id].x[i];
				int y = d_traces[agent_id].y[i];
				int qid = (y * COLS + x) * ACTIONS + (int)d_traces[agent_id].hist_actions[i];
				d_qtable[qid] += ALPHA * delta * d_traces[agent_id].hist_traces[i];
				d_traces[agent_id].hist_traces[i - 1] = GAMMA * LAMBDA * d_traces[agent_id].hist_traces[i];
			}

		}

		else {
			// memory shared 
			__shared__ float qval_cache[THREADS];

			unsigned int sid = threadIdx.x;
			unsigned int aid = sid & (ACTIONS - 1); // int aid = sid % ACTIONS;

													// next state
			qval_cache[sid] = d_qtable[nxt_qid + aid];

			// reduction, max qval
			unsigned int stride = ACTIONS >> 1; //ACTIONS / 2;

#pragma unroll
			while (stride != 0) {
				if (aid < stride) {
					if (qval_cache[sid] < qval_cache[sid + stride]) {
						qval_cache[sid] = qval_cache[sid + stride];
					}
				}
				__syncthreads();
				stride = stride >> 1; // stride /= 2;
			}

			if (aid == 0) { // if (sid &(ACTIONS - 1) == 0) { 
				d_qtable[cur_qid] += ALPHA * (rewards[agent_id] + GAMMA * qval_cache[sid] - d_qtable[cur_qid]);
			}
		}
	}
}


void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	dim3 block(THREADS, 1, 1);
	dim3 grid(NUM_AGENTS * ACTIONS / block.x, 1, 1);
	Agent_update << < grid, block >> > (cstate, nstate, rewards, d_qtable, d_action, d_active, d_traces);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////


/** CUDA Dynamic Parallelism
* @brief it will be called in __global__ Agent_action and Agent_update
*	for agent_id to calculate (.x) greedy_action and (.y) max_qval
*/

// struct actval {
// 	short action;
// 	float qval;
// }

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


/** Parallel Reduction for single-agent
*/

// <<< #agents, #actions >>>
// __global__ void Agent_action(int2 *cstate, short *d_action, curandState *d_state, float epsilon, float *d_qtable, bool *d_active) {

// 	// unsigned int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int agent_id = blockIdx.x;

// 	if (d_active[agent_id] == 1) 
// 	{
// 		// agent is alive 

// 		// located position on q_table
// 		unsigned int x = cstate[agent_id].x;
// 		unsigned int y = cstate[agent_id].y;


// 		float rand_state = curand_uniform(&d_state[agent_id]);
// 		short action;

// 		if (rand_state < epsilon) {
// 			// exploration
// 			action = (short)(curand_uniform(&d_state[agent_id]) * NUM_OF_ACTIONS);
// 			if (action == 4) {
// 				// curand_uniform (0, 1] for keeping uniform make the case action==4 as action==0
// 				action = 0; 
// 			}
// 		}
// 		else {
// 			// exploitation (greedy policy)

// 			// memory shared
// 			__shared__ float qval_cache[NUM_OF_ACTIONS]; // 4 actions  
//             __shared__ short action_cache[NUM_OF_ACTIONS];

//             unsigned int action_id = threadIdx.x;
//             action_cache[action_id] = (short)threadIdx.x;

//             unsigned int q_id = (y * COLS + x) * NUM_OF_ACTIONS;
//             qval_cache[action_id] = d_qtable[q_id + action_id];    

//             __syncthreads();

// 			// reduction for getting the max val and action
//             unsigned int i = blockDim.x / 2;

//             #pragma unroll
// 			while (i != 0) {
//                 if (action_id < i && qval_cache[action_id] < qval_cache[action_id + i])  {
//                     qval_cache[action_id] = qval_cache[action_id + i];
//                     action_cache[action_id] = action_cache[action_id + i];
//                 } 
//                 __syncthreads();
//                 i /= 2;
// 			} 
//             action = action_cache[0];
// 		}

// 		// decide the action
// 		d_action[agent_id] = action;
// 	}
// }


// short* agent_action(int2* cstate) {
// 	// do exploration or exploitation
// 	Agent_action <<<NUM_OF_AGENTS, NUM_OF_ACTIONS >>> (cstate, d_action, d_state, epsilon, d_qtable, d_active); 
// 	return d_action;
// }

////////////////////////////////////////////////////////////////////////////////////////////////////////
