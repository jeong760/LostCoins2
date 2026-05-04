/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdint.h>
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../Timer.h"

#include "GPUGroup.h"
#include "GPUMath.h"
#include "GPUHash.h"
#include "GPUBase58.h"
#include "GPUCompute.h"

// ---------------------------------------------------------------------------------------
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char* file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
	return;
}

// ---------------------------------------------------------------------------------------

__global__ void comp_keys(uint32_t mode, uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES,
	uint64_t* keys, uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeys(mode, keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, found);

}

__global__ void comp_keys_comp(uint8_t* bloomLookUp, int BLOOM_BITS, uint8_t BLOOM_HASHES, uint64_t* keys,
	uint32_t maxFound, uint32_t* found)
{

	int xPtr = (blockIdx.x * blockDim.x) * 8;
	int yPtr = xPtr + 4 * blockDim.x;
	ComputeKeysComp(keys + xPtr, keys + yPtr, bloomLookUp, BLOOM_BITS, BLOOM_HASHES, maxFound, found);

}

__global__ void clear_counter(uint32_t* found)
{
	ClearCouter(found);
}

// ---------------------------------------------------------------------------------------

using namespace std;

int _ConvertSMVer2Cores(int major, int minor)
{

	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{0x10, 	 8}, //Tesla Generation (SM 1.0) G80 class
		{0x11,  10}, //Tesla Generation (SM 1.1) G8x class
		{0x12,  11}, //Tesla Generation (SM 1.2) G9x class
		{0x13,  13}, // Tesla Generation (SM 1.3) GT200 class
		{0x20,  32}, // Fermi Generation (SM 2.0) GF100 class
		{0x21,  48}, // Fermi Generation (SM 2.1) GF10x class
		{0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
		{0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
		{0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
		{0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
		{0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
		{0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
		{0x60,  64}, // Pascal Generation (SM 6.0) GP100 class
		{0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
		{0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
		{0x70,  64}, // Volta Generation (SM 7.0) GV100 class
		{0x72,  64}, // Xavier Generation (SM 7.2) GV10B class
		{0x75,  64}, // Turing Generation (SM 7.5) TU100 class
		{0x80,  64}, // Ampere Generation (SM 8.0) A100 class
		{0x86, 128}, // Ampere Generation (SM 8.6) GeForce 30-series
		{0x87, 128}, // Ampere Generation (SM 8.7) RTX 30 or RTX A series
		{0x89, 128}, // Ada Lovelace Generation (SM 8.9) RTX40-series
		{0x90, 128}, // Hopper Generation (SM 9.0) H100 class
		{0xa0, 128}, // Blackwell Generation (SM 10.0) B100 class
		{0xa1, 128}, // Blackwell Generation (SM 10.1) B80 class
		{0xa2, 128}, // Blackwell Generation (SM 10.2)
		{0xb0, 128}, // Blackwell Tensor Core optimized
		{0xb1, 128}, // Blackwell variant
		{0xc0, 128}, // Blackwell
		{0xc1, 128}, // Blackwell
		{	-1, -1}
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// Default fallback for unknown architectures
	printf("Warning: Unknown GPU architecture SM %d.%d, using default core count\n", major, minor);
	return 128;

}

GPUEngine::GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound, bool rekey,
	int64_t BLOOM_SIZE, uint64_t BLOOM_BITS, uint8_t BLOOM_HASHES, const uint8_t* BLOOM_DATA,
	uint8_t* DATA, uint64_t TOTAL_ADDR)
{

	// Initialise CUDA
	this->rekey = rekey;
	this->nbThreadPerGroup = nbThreadPerGroup;

	this->BLOOM_SIZE = BLOOM_SIZE;
	this->BLOOM_BITS = BLOOM_BITS;
	this->BLOOM_HASHES = BLOOM_HASHES;
	this->DATA = DATA;
	this->TOTAL_ADDR = TOTAL_ADDR;

	initialised = false;

	int deviceCount = 0;
	CudaSafeCall(cudaGetDeviceCount(&deviceCount));

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("GPUEngine: There are no available device(s) that support CUDA\n");
		return;
	}

	// Validate GPU ID
	if (gpuId >= deviceCount) {
		printf("GPUEngine: Requested GPU %d but only %d device(s) available\n", gpuId, deviceCount);
		return;
	}

	CudaSafeCall(cudaSetDevice(gpuId));

	cudaDeviceProp deviceProp;
	CudaSafeCall(cudaGetDeviceProperties(&deviceProp, gpuId));

	// Optimize thread group size based on GPU architecture
	if (nbThreadGroup == -1) {
		// For newer architectures (compute capability >= 8.0), increase thread groups
		if (deviceProp.major >= 8) {
			nbThreadGroup = deviceProp.multiProcessorCount * 16;  // Increased for Ada/Hopper
		} else {
			nbThreadGroup = deviceProp.multiProcessorCount * 8;   // Original for older archs
		}
	}

	this->nbThread = nbThreadGroup * nbThreadPerGroup;
	this->maxFound = maxFound;
	this->outputSize = (maxFound * ITEM_SIZE + 4);

	char tmp[512];
	sprintf(tmp, "GPU #%d %s (%dx%d cores) Grid(%dx%d) [SM %d.%d]",
		gpuId, deviceProp.name, deviceProp.multiProcessorCount,
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		nbThread / nbThreadPerGroup,
		nbThreadPerGroup,
		deviceProp.major,
		deviceProp.minor);
	deviceName = std::string(tmp);

	// GPU-specific optimizations
	if (deviceProp.major == 8 && deviceProp.minor >= 6) {
		// Ampere (RTX 30 series)
		printf("Optimizing for Ampere architecture (RTX 30 series)\n");
		CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		CudaSafeCall(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
	}
	else if (deviceProp.major == 8 && deviceProp.minor == 9) {
		// Ada (RTX 40 series)
		printf("Optimizing for Ada architecture (RTX 40 series)\n");
		CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		CudaSafeCall(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
	}
	else if (deviceProp.major == 9) {
		// Hopper (H100)
		printf("Optimizing for Hopper architecture (H100)\n");
		CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		CudaSafeCall(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
	}
	else if (deviceProp.major >= 10) {
		// Blackwell
		printf("Optimizing for Blackwell architecture\n");
		CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
		CudaSafeCall(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
	}
	else {
		// Fallback for other architectures
		CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	}

	// Increase stack size for newer GPUs
	size_t stackSize = (deviceProp.major >= 8) ? 65536 : 49152;
	CudaSafeCall(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));

	// Check for unified memory support (compute capability >= 3.0)
	if (deviceProp.unifiedAddressing) {
		printf("GPU supports unified memory addressing\n");
	}

	// Allocate memory
	CudaSafeCall(cudaMalloc((void**)&inputKey, nbThread * 32 * 2));
	CudaSafeCall(cudaHostAlloc(&inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	CudaSafeCall(cudaMalloc((void**)&outputBuffer, outputSize));
	CudaSafeCall(cudaHostAlloc(&outputBufferPinned, outputSize, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	CudaSafeCall(cudaMalloc((void**)&inputBloomLookUp, BLOOM_SIZE));
	CudaSafeCall(cudaHostAlloc(&inputBloomLookUpPinned, BLOOM_SIZE, cudaHostAllocWriteCombined | cudaHostAllocMapped));

	memcpy(inputBloomLookUpPinned, BLOOM_DATA, BLOOM_SIZE);

	CudaSafeCall(cudaMemcpy(inputBloomLookUp, inputBloomLookUpPinned, BLOOM_SIZE, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaFreeHost(inputBloomLookUpPinned));
	inputBloomLookUpPinned = NULL;

	CudaSafeCall(cudaGetLastError());

	searchMode = SEARCH_COMPRESSED;
	searchType = P2PKH;
	initialised = true;

}

int GPUEngine::GetGroupSize()
{
	return GRP_SIZE;
}

void GPUEngine::PrintCudaInfo()
{
	const char* sComputeMode[] = {
		"Multiple host threads",
		"Only one host thread",
		"No host thread",
		"Multiple process threads",
		"Unknown",
		NULL
	};

	int deviceCount = 0;
	CudaSafeCall(cudaGetDeviceCount(&deviceCount));

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		printf("GPUEngine: There are no available device(s) that support CUDA\n");
		return;
	}

	printf("\n========== CUDA Device Information ==========\n\n");

	for (int i = 0; i < deviceCount; i++) {
		CudaSafeCall(cudaSetDevice(i));
		cudaDeviceProp deviceProp;
		CudaSafeCall(cudaGetDeviceProperties(&deviceProp, i));
		
		int cores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		int totalCores = cores * deviceProp.multiProcessorCount;
		
		printf("GPU #%d: %s\n", i, deviceProp.name);
		printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("  Total Cores: %d (%d MPs x %d cores/MP)\n", totalCores, deviceProp.multiProcessorCount, cores);
		printf("  Global Memory: %.1f GB\n", (double)deviceProp.totalGlobalMem / 1e9);
		printf("  Shared Memory/Block: %zu KB\n", deviceProp.sharedMemPerBlock / 1024);
		printf("  Max Threads/Block: %d\n", deviceProp.maxThreadsPerBlock);
		printf("  Warp Size: %d\n", deviceProp.warpSize);
		printf("  Max Grid Dimensions: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("  Unified Addressing: %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Compute Mode: %s\n", sComputeMode[deviceProp.computeMode]);
		printf("\n");
	}
	printf("============================================\n\n");
}

GPUEngine::~GPUEngine()
{
	CudaSafeCall(cudaFree(inputKey));
	CudaSafeCall(cudaFree(inputBloomLookUp));
	CudaSafeCall(cudaFreeHost(outputBufferPinned));
	CudaSafeCall(cudaFree(outputBuffer));
}

int GPUEngine::GetNbThread()
{
	return nbThread;
}

void GPUEngine::SetSearchMode(int searchMode)
{
	this->searchMode = searchMode;
}

void GPUEngine::SetSearchType(int searchType)
{
	this->searchType = searchType;
}

bool GPUEngine::callKernel()
{

	// Reset nbFound
	CudaSafeCall(cudaMemset(outputBuffer, 0, 4));

	// Call the kernel (Perform STEP_SIZE keys per thread)
	if (searchType == P2PKH) {
		if (searchMode == SEARCH_COMPRESSED) {
			comp_keys_comp << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
				(inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, maxFound, outputBuffer);
		}
		else {
			comp_keys << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
				(searchMode, inputBloomLookUp, BLOOM_BITS, BLOOM_HASHES, inputKey, maxFound, outputBuffer);
		}
	}
	else {
		printf("GPUEngine: Wrong searchType\n");
		return false;
	}

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
		return false;
	}
	return true;

}

bool GPUEngine::ClearOutBuffer()
{
	clear_counter << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> > (outputBuffer);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("GPUEngine: ClearOutBuffer: %s\n", cudaGetErrorString(err));
		return false;
	}
	return true;
}

bool GPUEngine::SetKeys(Point* p)
{

	// Sets the starting keys for each thread
	// p must contains nbThread public keys
	for (int i = 0; i < nbThread; i += nbThreadPerGroup) {
		for (int j = 0; j < nbThreadPerGroup; j++) {

			inputKeyPinned[8 * i + j + 0 * nbThreadPerGroup] = p[i + j].x.bits64[0];
			inputKeyPinned[8 * i + j + 1 * nbThreadPerGroup] = p[i + j].x.bits64[1];
			inputKeyPinned[8 * i + j + 2 * nbThreadPerGroup] = p[i + j].x.bits64[2];
			inputKeyPinned[8 * i + j + 3 * nbThreadPerGroup] = p[i + j].x.bits64[3];

			inputKeyPinned[8 * i + j + 4 * nbThreadPerGroup] = p[i + j].y.bits64[0];
			inputKeyPinned[8 * i + j + 5 * nbThreadPerGroup] = p[i + j].y.bits64[1];
			inputKeyPinned[8 * i + j + 6 * nbThreadPerGroup] = p[i + j].y.bits64[2];
			inputKeyPinned[8 * i + j + 7 * nbThreadPerGroup] = p[i + j].y.bits64[3];

		}
	}

	// Fill device memory
	CudaSafeCall(cudaMemcpy(inputKey, inputKeyPinned, nbThread * 32 * 2, cudaMemcpyHostToDevice));

	if (!rekey) {
		// We do not need the input pinned memory anymore
		CudaSafeCall(cudaFreeHost(inputKeyPinned));
		inputKeyPinned = NULL;
	}

	return callKernel();
}

bool GPUEngine::Launch(std::vector<ITEM>& dataFound, bool spinWait)
{
	dataFound.clear();


	// Get the result
	if (spinWait) {
		CudaSafeCall(cudaMemcpy(outputBufferPinned, outputBuffer, outputSize, cudaMemcpyDeviceToHost));
	}
	else {
		// Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy wich takes 100% CPU
		cudaEvent_t evt;
		CudaSafeCall(cudaEventCreate(&evt));
		CudaSafeCall(cudaMemcpyAsync(outputBufferPinned, outputBuffer, 4, cudaMemcpyDeviceToHost, 0));
		CudaSafeCall(cudaEventRecord(evt, 0));
		while (cudaEventQuery(evt) == cudaErrorNotReady) {
			// Sleep 1 ms to free the CPU
			Timer::SleepMillis(1);
		}
		CudaSafeCall(cudaEventDestroy(evt));
	}

	// Look for prefix found
	uint32_t nbFound = outputBufferPinned[0];
	if (nbFound > maxFound) {
		nbFound = maxFound;
	}

	// When can perform a standard copy, the kernel is eneded
	CudaSafeCall(cudaMemcpy(outputBufferPinned, outputBuffer, nbFound * ITEM_SIZE + 4, cudaMemcpyDeviceToHost));

	for (uint32_t i = 0; i < nbFound; i++) {

		uint32_t* itemPtr = outputBufferPinned + (i * ITEM_SIZE32 + 1);
		uint8_t* hash = (uint8_t*)(itemPtr + 2);
		if (CheckBinary(hash) > 0) {

			ITEM it;
			it.thId = itemPtr[0];
			int16_t* ptr = (int16_t*)&(itemPtr[1]);
			it.endo = ptr[0] & 0x7FFF;
			it.mode = (ptr[0] & 0x8000) != 0;
			it.incr = ptr[1];
			it.hash = (uint8_t*)(itemPtr + 2);
			dataFound.push_back(it);
		}
	}
	return callKernel();
}

int GPUEngine::CheckBinary(const uint8_t* hash)
{
	uint8_t* temp_read;
	uint64_t half, min, max, current; //, current_offset
	int64_t rcmp;
	int32_t r = 0;
	min = 0;
	current = 0;
	max = TOTAL_ADDR;
	half = TOTAL_ADDR;
	while (!r && half >= 1) {
		half = (max - min) / 2;
		temp_read = DATA + ((current + half) * 20);
		rcmp = memcmp(hash, temp_read, 20);
		if (rcmp == 0) {
			r = 1;  //Found!!
		}
		else {
			if (rcmp < 0) { //data < temp_read
				max = (max - half);
			}
			else { // data > temp_read
				min = (min + half);
			}
			current = min;
		}
	}
	return r;
}
