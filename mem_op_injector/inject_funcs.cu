/*
 * Copyright 2020, NVIDIA CORPORATION.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <stdint.h>
#include <stdio.h>

#include "nvbit_reg_rw.h"
#include "utils/utils.h"
#include "pf_injector.h"
#include "arch.h"


extern "C" __device__ __noinline__ void inject_error(uint64_t piinfo, uint64_t pverbose_device, int destGPRNum, int regval, int numDestGPRs, int maxRegs) {

				inj_info_t* inj_info = (inj_info_t*)piinfo; 
				uint32_t verbose_device = *((uint32_t *)pverbose_device);

				uint32_t smid;
				asm("mov.u32 %0, %smid;" :"=r"(smid));
				if (smid != inj_info->injSMID) 
								return; // This is not the selected SM. No need to proceed.

				uint32_t laneid;
				asm("mov.u32 %0, %laneid;" :"=r"(laneid));
				if (laneid != inj_info->injLaneID) 
								return; // This is not the selected Lane ID. No need to proceed.

				assert(numDestGPRs > 0);
				uint32_t injAfterVal = 0; 
				uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
				if (DUMMY) {
								injAfterVal = injBeforeVal;
				} else {
								injAfterVal = injBeforeVal ^ inj_info->injMask; 
								nvbit_write_reg(destGPRNum, injAfterVal);
				}
				// updating counter/flag to check whether the error was injected
				if (verbose_device) printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x\n", destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal);
				inj_info->errorInjected = true; 
				atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);  
}



extern "C" __device__ __noinline__ void clock_block(clock_t *d_o, clock_t clock_count)
{
    clock_t start_clock = clock();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count)
    {
        clock_offset = clock() - start_clock;
    }
     d_o[0] = clock_offset;
	 clock_t now = clock();
	//  printf("Clock start_clock %lu clock_offset %lu now %lu clock_count %lu\n", start_clock, clock_offset, now, clock_count);
}

extern "C" __device__ __noinline__ void print_op(uint64_t piinfo, uint64_t pverbose_device, int destGPRNum, int regval, int numDestGPRs, int maxRegs) {

				// printf("blockDim.x %d gridDim.x %d threadIdx.x %d blockIdx.x %d\n", blockDim.x, gridDim.x, threadIdx.x, blockIdx.x);
				inj_info_t* inj_info = (inj_info_t*)piinfo; 
				uint32_t verbose_device = *((uint32_t *)pverbose_device);
				clock_t cur = clock();
							
				assert(numDestGPRs > 0);
				clock_block(&cur, 1 0);
				//printf("Clock clock %lu \n", cur);
				inj_info->clock = cur;
				// uint32_t smid;
				// asm("mov.u32 %0, %smid;" :"=r"(smid));
				// if (smid != inj_info->injSMID) 
				// 				return; // This is not the selected SM. No need to proceed.

				// uint32_t laneid;
				// asm("mov.u32 %0, %laneid;" :"=r"(laneid));
				// if (laneid != inj_info->injLaneID) 
				//              return; // This is not the selected Lane ID. No need to proceed.

				// printf("Clock clock %lu \n", cur);
				// uint32_t injAfterVal = 0; 
				// uint32_t injBeforeVal = nvbit_read_reg(destGPRNum); // read the register value
				// if (DUMMY) {
				// 				injAfterVal = injBeforeVal;
				// } else {
				// 				injAfterVal = injBeforeVal ^ inj_info->injMask; 
				// 				nvbit_write_reg(destGPRNum, injAfterVal);
				// }
				// updating counter/flag to check whether the error was injected
				// if (verbose_device) printf("register=%d, before=0x%x, after=0x%x, expected_after=0x%x instType=%lu\n", destGPRNum, injBeforeVal, nvbit_read_reg(destGPRNum), injAfterVal, inj_info->injInstType);
				// inj_info->errorInjected = true; 
				// atomicAdd((unsigned long long*) &inj_info->injNumActivations, 1LL);  
}

