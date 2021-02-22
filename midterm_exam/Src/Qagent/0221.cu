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

short *d_action;
curandState *d_state;

bool  *d_active;
float *d_qtable;

float epsilon;


//////////////////////////	agent_init()  //////////////////////////

// <<< 1, #agents >>>
__global__ void Init_agent(curandState *d_state, bool *d_active) 
{
	unsigned int agent_id = threadIdx.x + blockIdx.x * blockDim.x;

	curand_init(clock() + agent_id, agent_id, 0, &d_state[agent_id]);
	d_active[agent_id] = 1;
}


// <<< (#cols, #rows), #actions >>>
__global__ void Init_qtable(float *d_qtable) 
{    
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int nx = COLS * NUM_OF_ACTIONS;
	unsigned int tid = iy * nx + ix;
	
    d_qtable[tid] = 0;
}


void agent_init()
{
	// clear action + initQ table + self initialization

	epsilon = EPSILON;
	cudaMalloc((void **)&d_action, sizeof(short) * NUM_OF_AGENTS);

	cudaMalloc((void **)&d_state, sizeof(curandState) * NUM_OF_AGENTS);
	cudaMalloc((void **)&d_active, sizeof(bool) * NUM_OF_AGENTS);
    
	Init_agent << <1, NUM_OF_AGENTS >> > (d_state, d_active);

	unsigned int qSize = NUM_OF_ACTIONS * COLS * ROWS;
	cudaMalloc((void **)&d_qtable, sizeof(float) * qSize);

	dim3 grid(COLS, ROWS);
	dim3 block(NUM_OF_ACTIONS);
	Init_qtable << <grid, block >> > (d_qtable);

}


//////////////////////////	agent_init_episode() //////////////////////////

// <<< 1, #agents >>> 
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

// <<< #agents, #actions >>>
__global__ void Agent_action(int2 *cstate, short *d_action, curandState *d_state, float epsilon, float *d_qtable, bool *d_active) {
    
	// unsigned int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int agent_id = blockIdx.x;

	if (d_active[agent_id] == 1) 
	{
		// agent is alive 

		// located position on q_table
		unsigned int x = cstate[agent_id].x;
		unsigned int y = cstate[agent_id].y;

		
		float rand_state = curand_uniform(&d_state[agent_id]);
		short action;

		if (rand_state < epsilon) {
			// exploration
			action = (short)(curand_uniform(&d_state[agent_id]) * NUM_OF_ACTIONS);
			if (action == 4) {
				// curand_uniform (0, 1] for keeping uniform make the case action==4 as action==0
				action = 0; 
			}
		}
		else {
			// exploitation (greedy policy)

			// memory shared
			__shared__ float qval_cache[NUM_OF_ACTIONS]; // 4 actions  
            __shared__ short action_cache[NUM_OF_ACTIONS];

            unsigned int action_id = threadIdx.x;
            action_cache[action_id] = (short)threadIdx.x;

            unsigned int q_id = (y * COLS + x) * NUM_OF_ACTIONS;
            qval_cache[action_id] = d_qtable[q_id + action_id];    
            
            __syncthreads();

			// reduction for getting the max val and action
            unsigned int i = blockDim.x / 2;

            #pragma unroll
			while (i != 0) {
                if (action_id < i && qval_cache[action_id] < qval_cache[action_id + i])  {
                    qval_cache[action_id] = qval_cache[action_id + i];
                    action_cache[action_id] = action_cache[action_id + i];
                } 
                __syncthreads();
                i /= 2;
			} 
            action = action_cache[0];
		}

		// decide the action
		d_action[agent_id] = action;
	}
}


short* agent_action(int2* cstate) {
	// do exploration or exploitation
	Agent_action <<<NUM_OF_AGENTS, NUM_OF_ACTIONS >>> (cstate, d_action, d_state, epsilon, d_qtable, d_active); 
	return d_action;
}


//////////////////////////	agent_update() //////////////////////////

// <<< #agents, #actions >>>
__global__ void Agent_update(int2* cstate, int2* nstate, float *rewards, float *d_qtable, short *d_action, bool *d_active)
{
	// observe next state S' and R
    unsigned int agent_id = blockIdx.x;

	if (d_active[agent_id] == 1) {
		// agent active
		unsigned int x0 = cstate[agent_id].x;
		unsigned int y0 = cstate[agent_id].y;

		unsigned int x1 = nstate[agent_id].x;
		unsigned int y1 = nstate[agent_id].y;

		float gamma_item = 0; // if agent is inactive, the gamma_item == 0

		if (rewards[agent_id] == 0) {
			// agent still active

			// memory shared
            __shared__ float qval_cache[NUM_OF_ACTIONS];
            unsigned int action_id = threadIdx.x;

			unsigned int n_qid = (y1 * COLS + x1) * NUM_OF_ACTIONS; // next state (n+1)
			qval_cache[action_id] = d_qtable[n_qid + action_id];

            __syncthreads();

            // reduction
			unsigned int i = blockDim.x / 2;

            #pragma unroll
            while (i != 0) {
                if (action_id < i && qval_cache[action_id] < qval_cache[action_id + i]) {
                    qval_cache[action_id] = qval_cache[action_id + i];
                }
                __syncthreads();
                i /= 2;
            }

            float best_next_qval = qval_cache[0];
			gamma_item = GAMMA * best_next_qval;

		}

		// update q_table of current state (n) <- max val of next state (n+1)
		// Q(S, A) <- Q(S, A) + alpha[R + gamma * max Q(S', a) - Q(S, A)]

		unsigned int c_qid = (y0 * COLS + x0) * NUM_OF_ACTIONS + (int)d_action[agent_id];
		d_qtable[c_qid] += ALPHA * (rewards[agent_id] + gamma_item - d_qtable[c_qid]);

	}
}


void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	Agent_update <<<NUM_OF_AGENTS, NUM_OF_ACTIONS >>> (cstate, nstate, rewards, d_qtable, d_action, d_active);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
