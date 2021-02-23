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

// #mines=96, #flag=1

#define COLS	32
#define ROWS	32

#define NUM_OF_AGENTS	128
#define NUM_OF_ACTIONS	4

#define GAMMA		0.9
#define ALPHA		0.1
#define	EPSILON		1.0
#define DELTA_EPS	0.0001

short *d_action;
curandState *d_state;
bool  *d_active;
float *d_qtable;

float epsilon;
float *d_epsilon;

/**	Host: 	agent_init()  ////////////////////////////////////////////
* @brief 	clear action + initQ table + self initialization
*/

__global__ void Init_agent(curandState *d_state, bool *d_active) {

	int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
	// d_action[agent_id] = 2;
	curand_init(clock() + agent_id, agent_id, 0, &d_state[agent_id]);
	d_active[agent_id] = 1;
}

__global__ void Init_epsilon(float *d_epsilon) {
	*d_epsilon = 1.0000f;
}

__global__ void Init_qtable(float *d_qtable) {
	// it's 4 x board_size, as the qtable includes 4 actions at each position.
	// init Q-table Q(s, a) = 0, s in S, a in A(s)
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int tid = iy * (COLS * NUM_OF_ACTIONS) + ix;
	d_qtable[tid] = 0;
}

void agent_init()
{
	// clear action + initQ table + self initialization
	cudaMalloc((void **)&d_action, sizeof(short) * NUM_OF_AGENTS);

	cudaMalloc((void **)&d_state, sizeof(curandState) * NUM_OF_AGENTS);
	cudaMalloc((void **)&d_active, sizeof(bool) * NUM_OF_AGENTS);
	Init_agent << <1, NUM_OF_AGENTS >> > (d_state, d_active);

	cudaMalloc((void **)&d_epsilon, sizeof(float));
	Init_epsilon << <1, 1 >> > (d_epsilon);

	int qSize = NUM_OF_ACTIONS * COLS * ROWS;
	cudaMalloc((void **)&d_qtable, sizeof(float) * qSize);
	dim3 grid(COLS, ROWS);
	dim3 block(NUM_OF_ACTIONS);
	Init_qtable << <grid, block >> > (d_qtable);

}

/** Host:	agent_init_episode() //////////////////////////////////////////////
* @brief 	set all agents in active status
*/

__global__ void Init_epsiode(bool *d_active) {
	// agent 1 alive, 0 dead;
	int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
	d_active[agent_id] = 1;
}

void agent_init_episode() {
	// set all agents in active status
	Init_epsiode << <1, NUM_OF_AGENTS >> > (d_active);
}

/** Host: 	adjust_epsilon() ////////////////////////////////////////////
* @brief 	adjust epsilon, return a CPU variable
*/

__global__ void Adjust_epsilon(float *d_epsilon) {
	if (*d_epsilon > 1.0000f) {
		*d_epsilon = 1.0000f;
	}
	else if (*d_epsilon < 0.1000f) {
		*d_epsilon = 0.1000f;
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

/** Host:	agent_action() ////////////////////////////////////////////
* @brief	if agent is alive, run algorithm to take action
*/


__global__ void Agent_action(int2 *cstate, short *d_action, curandState *d_state, float *d_epsilon, float *d_qtable, bool *d_active) {

	int agent_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (d_active[agent_id] == 1) {
		// agent is alive 
		int x = cstate[agent_id].x;
		int y = cstate[agent_id].y;

		// the x, y coodinate will be block.x block.y in qtable
		// for each pos, there are still 4 action types;

		// exploration
		float rand_state = curand_uniform(&d_state[agent_id]);
		short action;
		if (rand_state < *d_epsilon) {
			// float div = 1.000f / ((float)NUM_OF_ACTIONS);
			action = (short)(curand_uniform(&d_state[agent_id]) * NUM_OF_ACTIONS); 
		}
		else {
			// exploitation (greedy policy)
			int qid = (y * COLS + x) * NUM_OF_ACTIONS;
			// int qid = y * (COLS * NUM_OF_ACTIONS) + (x * NUM_OF_ACTIONS);
			float max_qval = d_qtable[qid];
			action = 0;
			for (int i = 1; i < NUM_OF_ACTIONS; ++i) {
				if (d_qtable[qid + i] > max_qval) {
					max_qval = d_qtable[qid + i];
					action = (short)i;
				}
			}
		}

		// decide the action
		d_action[agent_id] = action;
	}
}


short* agent_action(int2* cstate) {
	// do exploration or exploitation
	Agent_action << <1, NUM_OF_AGENTS >> > (cstate, d_action, d_state, d_epsilon, d_qtable, d_active); // , d_active
	return d_action;
}

/** Host:	agent_update() ////////////////////////////////////////////
* @brief	if agent is alive, update qtable 
*/

__global__ void Agent_update(int2* cstate, int2* nstate, float *rewards, float *d_qtable, short *d_action, bool *d_active)
{
	// observe next state S' and R
	int agent_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (d_active[agent_id] == 1) {
		// agent active
		int x0 = cstate[agent_id].x;
		int y0 = cstate[agent_id].y;

		int x1 = nstate[agent_id].x;
		int y1 = nstate[agent_id].y;

		float gamma_item = 0;

		if (rewards[agent_id] == 0) {
			// next state (n+1)
			int n_qid = (y1 * COLS + x1) * NUM_OF_ACTIONS;
			// int n_qid = (y1 * COLS * NUM_OF_ACTIONS) + (x1 * NUM_OF_ACTIONS);
			float best_next_qval = d_qtable[n_qid];
			// i start from 1 as the i = 0 has been assign as init max_qval
			for (int i = 1; i < NUM_OF_ACTIONS; ++i) {
				if (d_qtable[n_qid + i] > best_next_qval) {
					best_next_qval = d_qtable[n_qid + i];
				}
			}
			gamma_item = GAMMA * best_next_qval;
			// agent still active
		}

		// update q_table of current state (n) by max val of next state
		// Q(S, A) <- Q(S, A) + alpha[R + gamma * max Q(S', a) - Q(S, A)]
		int c_qid = (y0 * COLS + x0) * NUM_OF_ACTIONS + (int)d_action[agent_id];
		// int c_qid = (y0 * COLS * NUM_OF_ACTIONS) + (x0 * NUM_OF_ACTIONS) + (int)d_action[agent_id];
		d_qtable[c_qid] += ALPHA * (rewards[agent_id] + gamma_item - d_qtable[c_qid]);

		// update state to next
		if (rewards[agent_id] == 0) {
			// agent status: active
			cstate[agent_id] = nstate[agent_id];
		}
		else {
			// agent status: inactive 
			cstate[agent_id].x = 0;
			cstate[agent_id].y = 0;
			d_active[agent_id] = 0;
		}
	}
}


void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	// add your codes
	Agent_update << <1, NUM_OF_AGENTS >> > (cstate, nstate, rewards, d_qtable, d_action, d_active);
}



