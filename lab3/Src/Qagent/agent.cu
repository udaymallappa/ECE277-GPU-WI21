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

float epsilon;
short *d_action;

void agent_init()
{
	// add your codes
}

void agent_init_episode() 
{
	// add your codes
}

float agent_adjustepsilon() 
{
	// add your codes

	return epsilon;
}

short* agent_action(int2* cstate)
{
	// add your codes

	return d_action;
}

void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	// add your codes
}
