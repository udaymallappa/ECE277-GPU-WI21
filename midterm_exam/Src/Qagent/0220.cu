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


#define COLS	46
#define ROWS	46

#define NUM_OF_AGENTS	512
#define NUM_OF_ACTIONS	4

#define GAMMA		0.9
#define ALPHA		0.5
#define	EPSILON		1.0
#define EPS_CEIL	1.0
#define EPS_BOTTOM  0.0
#define DELTA_EPS	0.01

short *d_action;        // sizeof * agents
curandState *d_state;   // sizeof * agents
bool  *d_active;        // sizeof * agents
float4 *d_qtable;       // sizeof * cols * rows: .x .y .z .w actions
float2 *d_qmax;         // sizeof * cols * rows: .x best action , .y max qval

float epsilon;


//////////////////////////	agent_init()  //////////////////////////


__global__ void Init_agent(curandState *d_state, bool *d_active) {

	unsigned int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(clock() + agent_id, agent_id, 0, &d_state[agent_id]);
	d_active[agent_id] = 1;
}


__global__ void Init_qtable(float4 *d_qtable, float2 *d_qmax) {

	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

	d_qtable[tid].x = 0;
	d_qtable[tid].y = 0;
	d_qtable[tid].z = 0;
	d_qtable[tid].w = 0;

    d_qmax[tid].x = 0;
    d_qmax[tid].y = 0;
}

void agent_init()
{
	epsilon = EPSILON;
	// clear action + initQ table + self initialization
	cudaMalloc((void **)&d_action, sizeof(short) * NUM_OF_AGENTS);
	cudaMalloc((void **)&d_state, sizeof(curandState) * NUM_OF_AGENTS);
	cudaMalloc((void **)&d_active, sizeof(bool) * NUM_OF_AGENTS);
	Init_agent << <1, NUM_OF_AGENTS >> > (d_state, d_active);

	unsigned int tableSize = COLS * ROWS;
	cudaMalloc((void **)&d_qtable, sizeof(float4) * tableSize);
    cudaMalloc((void **)&d_qmax, sizeof(float2) * tableSize);
	Init_qtable << <COLS, ROWS >> > (d_qtable, d_qmax);

}

//////////////////////////	agent_init_episode() //////////////////////////


__global__ void Init_epsiode(bool *d_active) {
	// agent 1 alive, 0 dead;
	unsigned int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
	d_active[agent_id] = 1;
}

void agent_init_episode() {
	// set all agents in active status
	Init_epsiode << <1, NUM_OF_AGENTS >> > (d_active);
}

//////////////////////////	adjust_epsilon() //////////////////////////


float agent_adjustepsilon()
{
	if (epsilon > EPS_CEIL) {
		epsilon = EPS_CEIL;
	} else if (epsilon < EPS_BOTTOM) {
		epsilon = EPS_BOTTOM;
	} else {
		epsilon -= DELTA_EPS;
	}
	return epsilon;
}

//////////////////////////	agent_action() //////////////////////////


__global__ void Agent_action(int2 *cstate, short *d_action, curandState *d_state, float epsilon, float4 *d_qtable, bool *d_active, float2 *d_qmax) {

	unsigned int agent_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (d_active[agent_id] == 1) {
		// agent is alive 
		unsigned int x = cstate[agent_id].x;
		unsigned int y = cstate[agent_id].y;

		// exploration
		float rand_state = curand_uniform(&d_state[agent_id]);
		float action;
		if (rand_state < epsilon) {
			action = (curand_uniform(&d_state[agent_id]) * NUM_OF_ACTIONS);
			if (action == 4.0f) action = 0.0f; // curand_uniform (0, 1]
		}
		else {
			// exploitation (greedy policy)
            unsigned int tid = y * COLS + x;
            action = d_qmax[tid].x;
		}
		// decide the action
		d_action[agent_id] = (short)action;
	}
}


short* agent_action(int2* cstate) {
	// do exploration or exploitation
	Agent_action << <1, NUM_OF_AGENTS >> > (cstate, d_action, d_state, epsilon, d_qtable, d_active, d_qmax); 
	return d_action;
}

//////////////////////////	agent_update() //////////////////////////


__global__ void Agent_update(int2* cstate, int2* nstate, float *rewards, float4 *d_qtable, short *d_action, bool *d_active, float2 *d_qmax)
{
	// observe next state S' and R
	unsigned int agent_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (d_active[agent_id] == 1) {
		// agent active
		unsigned int x0 = cstate[agent_id].x;
		unsigned int y0 = cstate[agent_id].y;

		unsigned int x1 = nstate[agent_id].x;
		unsigned int y1 = nstate[agent_id].y;

		float gamma_item = 0;

		if (rewards[agent_id] == 0) {
			// agent still active
			// unsigned int n_qid = (y1 * COLS + x1) * NUM_OF_ACTIONS; // next state (n+1)
			// float best_next_qval = d_qtable[n_qid];
			// // i start from 1 as the i = 0 has been assign as init max_qval
			// for (int i = 1; i < NUM_OF_ACTIONS; ++i) {
			// 	if (d_qtable[n_qid + i] > best_next_qval) {
			// 		best_next_qval = d_qtable[n_qid + i];
			// 	}
			// }
            unsigned int n_tid = y1 * COLS + x1;
            float best_next_qval = d_qmax[n_tid].y;
			gamma_item = GAMMA * best_next_qval;
		}

		// update q_table of current state (n) by max val of next state
		// Q(S, A) <- Q(S, A) + alpha[R + gamma * max Q(S', a) - Q(S, A)]
        unsigned int c_tid = y0 * COLS + x0;
		// unsigned int c_qid = (y0 * COLS + x0) * NUM_OF_ACTIONS + (int)d_action[agent_id];
        if (d_action[agent_id] == 0) {
            d_qtable[c_tid].x += ALPHA * (rewards[agent_id] + gamma_item - d_qtable[c_tid].x);
            if (d_qtable[c_tid].x > d_qmax[c_tid].y) {
                d_qmax[c_tid].y = d_qtable[c_tid].x;
                d_qmax[c_tid].x = 0.0f;
            }
        } else if (d_action[agent_id] == 1) {
            d_qtable[c_tid].y += ALPHA * (rewards[agent_id] + gamma_item - d_qtable[c_tid].y);
            if (d_qtable[c_tid].y > d_qmax[c_tid].y) {
                d_qmax[c_tid].y = d_qtable[c_tid].y;
                d_qmax[c_tid].x = 1.0f;
            }
        } else if (d_action[agent_id] == 2) {
            d_qtable[c_tid].z += ALPHA * (rewards[agent_id] + gamma_item - d_qtable[c_tid].z);
            if (d_qtable[c_tid].z > d_qmax[c_tid].y) {
                d_qmax[c_tid].y = d_qtable[c_tid].x;
                d_qmax[c_tid].x = 2.0f;
            }
        } else if (d_action[agent_id] == 3) {
            d_qtable[c_tid].w += ALPHA * (rewards[agent_id] + gamma_item - d_qtable[c_tid].w);
            if (d_qtable[c_tid].w > d_qmax[c_tid].y) {
                d_qmax[c_tid].y = d_qtable[c_tid].w;
                d_qmax[c_tid].x = 3.0f;
            }
        }

		// d_qtable[c_qid] += ALPHA * (rewards[agent_id] + gamma_item - d_qtable[c_qid]);
	}
}


void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	Agent_update << <1, NUM_OF_AGENTS >> > (cstate, nstate, rewards, d_qtable, d_action, d_active, d_qmax);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////

// float *d_epsilon;

			// float div = 1.000f / ((float)NUM_OF_ACTIONS);

// __global__ void Init_epsilon(float *d_epsilon) {
// 	*d_epsilon = 1.0f;
// }


	// cudaMalloc((void **)&d_epsilon, sizeof(float));
	// Init_epsilon << <1, 1 >> > (d_epsilon);

			// else {
		// 	// agent status: inactive 
		// 	cstate[agent_id].x = 0;
		// 	cstate[agent_id].y = 0;
		// 	d_active[agent_id] = 0;
		// }

		// Adjust_epsilon << <1, 1 >> > (&epsilon);
	// cudaMemcpy(&epsilon, d_epsilon, sizeof(float), cudaMemcpyDeviceToHost);

	
// __global__ void Adjust_epsilon(float &epsilon) { // float *d_epsilon
// 	if (epsilon > 1.0f) {
// 		epsilon = 1.0f;
// 	}
// 	else if (epsilon < 0.0f) {
// 		epsilon = 0.0f;
// 	}
// 	else {
// 		epsilon -= DELTA_EPS;
// 	}
// }

			// int n_qid = (y1 * COLS * NUM_OF_ACTIONS) + (x1 * NUM_OF_ACTIONS);
			// int qid = y * (COLS * NUM_OF_ACTIONS) + (x * NUM_OF_ACTIONS);
		// int c_qid = (y0 * COLS * NUM_OF_ACTIONS) + (x0 * NUM_OF_ACTIONS) + (int)d_action[agent_id];

		
		// the x, y coodinate will be block.x block.y in qtable
		// for each pos, there are still 4 action types;

			// // update state to next
		// if (rewards[agent_id] == 0) {
		// 	// agent status: active
		// 	cstate[agent_id] = nstate[agent_id];
		// }