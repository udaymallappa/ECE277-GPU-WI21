/*************************************************************************
/* ECE 277: GPU Programmming 2021 WINTER quarter
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


#define ACTIONS		4
#define NUM_AGENTS	512

#define COLS		46
#define ROWS		46
#define QSIZE 		COLS * ROWS * ACTIONS

#define THREADS		1024
#define WARPSIZE	32

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

// float2 *d_actval; 	// sizeof() * #agents : .x action, .y qval; 

float epsilon;


//////////////////////////	agent_init()  //////////////////////////

// <<< NUM_AGENTS 512 / WARPSIZE 32 = 16, WARPSIZE 32 = 32 >>>
__global__ void Init_agent(curandState *d_states, bool *d_active)
{
	unsigned int agent_id = threadIdx.x + blockIdx.x * blockDim.x;

	curand_init(clock() + agent_id, agent_id, 0, &d_states[agent_id]);
	d_active[agent_id] = 1;
}


// <<< (#cols 46/2, #rows 46/2), #actions 4*4 >>> (23, 23), (16)
// occupency
__global__ void Init_qtable(float *d_qtable)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int nx = gridDim.x * blockDim.x;
	unsigned int tid = ix + iy * nx; 
	d_qtable[tid] = 0.0f;
}


void agent_init()
{
	// clear action + initQ table + self initialization
	epsilon = EPSILON;

	cudaMalloc((void **)&d_action, NUM_AGENTS * sizeof(short));
	// cudaMalloc((void **)&d_actval, NUM_AGENTS * sizeof(float2)); // used in __device__ void

	cudaMalloc((void **)&d_states, NUM_AGENTS * sizeof(curandState));
	cudaMalloc((void **)&d_active, NUM_AGENTS * sizeof(bool));

	dim3 block(WARPSIZE, 1, 1);
	dim3 grid(NUM_AGENTS / WARPSIZE, 1, 1);
	// dim3 block(512, 1, 1), dim3 grid(1, 1, 1);
	Init_agent << <grid, block >> > (d_states, d_active);

	cudaMalloc((void **)&d_qtable, QSIZE * sizeof(float));

	dim3 qblock(ACTIONS * 4, 1, 1);
	dim3 qgrid(COLS / 2, ROWS / 2, 1);
	// dim3 qblock(4, 1, 1), dim3 qgrid(46, 46, 1);
	Init_qtable << < qgrid, qblock >> > (d_qtable);		// <<< (23, 23), 16 >>>

}


//////////////////////////	agent_init_episode() //////////////////////////

// <<< NUM_AGENTS 512 / WARPSIZE = 16, WARPSIZE = 32 >>>
__global__ void Init_epsiode(bool *d_active) {
	// agent 1 alive, 0 dead;
	unsigned int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
	d_active[agent_id] = 1;
}


void agent_init_episode() {
	// set all agents in active status
	dim3 block(WARPSIZE, 1, 1);
	dim3 grid(NUM_AGENTS / WARPSIZE, 1, 1);
	// dim3 block(512, 1, 1);
	// dim3 grid(1, 1, 1);
	Init_epsiode << <grid, block >> > (d_active);
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
		epsilon -= DELTA_EPS;
	}
	return epsilon;
}


//////////////////////////	agent_action() //////////////////////////




// <<< NUM_AGENTS 512 * ACTIONS 4 / THREADS = 2 , THREADS = 1024 >>>
__global__ void Agent_action(int2 *cstate, short *d_action, curandState *d_states, float epsilon, float *d_qtable, bool *d_active)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int agent_id = tid / ACTIONS;
	// unsigned int agent_id = blockIdx.x; 

	if (d_active[agent_id] == 1)
	{
		// agent is alive 
		float rand_state = curand_uniform(&d_states[agent_id]);

		if (rand_state < epsilon) {
			// exploration
			short action = (short)(curand_uniform(&d_states[agent_id]) * ACTIONS);
			if (action == 4) {
				// curand_uniform (0, 1] for keeping uniform make the case action==4 as action==0
				action = 0;
			}
			d_action[agent_id] = action;
		}
		else {
			// exploitation (greedy policy)
			// Get_qAction_qMaxVal << <1, ACTIONS >> > (cstate, d_qtable, d_actval, agent_id);

			unsigned int x = cstate[agent_id].x;
			unsigned int y = cstate[agent_id].y;

			unsigned int sid = threadIdx.x;  	// 0123 4567 .. ...1023

			// extern __shared__ float qval_cache[]; 
			// extern __shared__ short action_cache[];

			__shared__ float qval_cache[THREADS]; 
			__shared__ short action_cache[THREADS];

			// unsigned int aid = tid - agent_id * ACTIONS;
			unsigned int aid = sid % 4;
			action_cache[sid] = (short)aid;  // 0123 0123 .. ...3 

			unsigned int q_id = (y * COLS + x) * ACTIONS;
			qval_cache[sid] = d_qtable[q_id + aid];  // 

			__syncthreads();

			unsigned int stride = ACTIONS / 2; 

			#pragma unroll
			while (stride != 0) {
				// 1st round, stride==2 : 01(23), 0<-2  1<-3
				// 2nd round, stride==1 : 0(123), 0<-1
				if (aid < stride && qval_cache[sid] < qval_cache[sid + stride]) {
					// keep larger values in left cache
					qval_cache[sid] = qval_cache[sid + stride];
					action_cache[sid] = action_cache[sid + stride];
				}
				__syncthreads();
				stride /= 2;
			}
			// update: .x action; .y max_qval.
			// d_actval[agent_id].x = action_cache[agent_id * ACTIONS];
			// d_actval[agent_id].y = (float)qval_cache[agent_id * ACTIONS];
			// d_action[agent_id] = action_cache[agent_id * ACTIONS];
			d_action[agent_id] = action_cache[sid / ACTIONS];
			// d_action[agent_id] = (short)d_actval[agent_id].x; // .x action .y max_qval
		}
		// decide the action
		// d_action[agent_id] = action;
	}
}


short* agent_action(int2* cstate) {
	// do exploration or exploitation
	dim3 block(THREADS, 1, 1);
	dim3 grid(NUM_AGENTS * ACTIONS / THREADS, 1, 1);
	// dim3 block(4, 1, 1);
	// dim3 grid(512, 1, 1);
	// <<< NUM_AGENTS 512 * ACTIONS 4 / THREADS = 2 , THREADS = 1024 >>>
	Agent_action <<< grid, block >>> (cstate, d_action, d_states, epsilon, d_qtable, d_active);
	return d_action;
}


//////////////////////////	agent_update() //////////////////////////

// <<< NUM_AGENTS 512 * ACTIONS 4 / THREADS = 2 , THREADS = 1024 >>>
__global__ void Agent_update(int2* cstate, int2* nstate, float *rewards, float *d_qtable, short *d_action, bool *d_active)
{
	// observe next state S' and R
	// unsigned int agent_id = threadIdx.y;
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// unsigned int agent_id = tid / ACTIONS;
	unsigned int agent_id = blockIdx.x;

	if (d_active[agent_id] == 1) {
		// agent active
		float gamma_item = 0; // if agent is inactive, the gamma_item == 0

		if (rewards[agent_id] == 0) {
			// agent still active
			// Get_qAction_qMaxVal << < 1, ACTIONS >> > (nstate, d_qtable, d_actval, agent_id);

			unsigned int x = nstate[agent_id].x;
			unsigned int y = nstate[agent_id].y;		

			// extern __shared__ float qval_cache[]; 
			// extern __shared__ short action_cache[];

			__shared__ float qval_cache[4]; 
			__shared__ short action_cache[4];

			unsigned int aid = tid - agent_id * ACTIONS;
			action_cache[aid] = aid;  // 0123 .. 0123 0123 ..

			unsigned int q_id = (y * COLS + x) * ACTIONS;
			qval_cache[aid] = d_qtable[q_id + aid];  // 

			__syncthreads();

			unsigned int stride = ACTIONS / 2; 

			#pragma unroll
			while (stride != 0) {
				// 1st round, stride==2 : 01(23), 0<-2  1<-3
				// 2nd round, stride==1 : 0(123), 0<-1
				if (aid < stride && qval_cache[aid] < qval_cache[aid + stride]) {
					// keep larger values in left cache
					qval_cache[aid] = qval_cache[aid + stride];
					action_cache[aid] = action_cache[aid + stride];
				}
				__syncthreads();
				stride /= 2;
			}
			// update: .x action; .y max_qval.
			// d_actval[agent_id].x = action_cache[agent_id * ACTIONS];
			// d_actval[agent_id].y = qval_cache[agent_id * ACTIONS];

			// float best_next_qval = d_actval[agent_id].y; // .x action .y max_qval
			float best_next_qval = qval_cache[0];
			gamma_item = GAMMA * best_next_qval; //qval_cache[agent_id * ACTIONS]; // max qval for next state
		}

		// update q_table of current state (n) <- max val of next state (n+1)
		// Q(S, A) <- Q(S, A) + alpha[R + gamma * max Q(S', a) - Q(S, A)]

		unsigned int x0 = cstate[agent_id].x;
		unsigned int y0 = cstate[agent_id].y;

		unsigned int c_qid = (y0 * COLS + x0) * ACTIONS + (unsigned int)d_action[agent_id];
		d_qtable[c_qid] += ALPHA * (rewards[agent_id] + gamma_item - d_qtable[c_qid]);
	}
}


void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	// dim3 block(THREADS, 1, 1);
	// dim3 grid(NUM_AGENTS * ACTIONS / THREADS, 1, 1);
	dim3 block(4, 1, 1);
	dim3 grid(512, 1, 1);
	// <<< NUM_AGENTS 512 * ACTIONS 4 / THREADS = 2 , THREADS = 1024 >>>
	Agent_update <<< grid, block >>> (cstate, nstate, rewards, d_qtable, d_action, d_active);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////


/** <<< 1, #actions 4 >>>  CUDA Dynamic Parallelism
* @brief it will be called in __global__ Agent_action and Agent_update
* 		  for agent_id to calculate (.x) greedy_action and (.y) max_qval
* @param cstate 	int2
* @param d_qtable 	float
* @param d_actval 	float2
* @param agent_id 	unsigned int
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

// 	#pragma unroll
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