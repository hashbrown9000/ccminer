extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_fugue.h"
}

#include "miner.h"
#include "cuda_helper.h"

#include <stdio.h>
#include <memory.h>

//#define DR_TIMINGS

//#define DROP_OFFF

#ifdef DROP_OFFF
static uint16_t* d_poks[MAX_GPUS];
#endif

static uint32_t* d_hash[MAX_GPUS];
static uint64_t* d_roundInfo[MAX_GPUS];

extern "C" static void switch_hash(const void *input, void *output, int id)
{
	switch (id) {
	case 0:
		sph_keccak512_context ctx_keccak;
		sph_keccak512_init(&ctx_keccak);
		sph_keccak512(&ctx_keccak, input, 64);
		sph_keccak512_close(&ctx_keccak, output);
		break;
	case 1:
		sph_blake512_context ctx_blake;
		sph_blake512_init(&ctx_blake);
		sph_blake512(&ctx_blake, input, 64);
		sph_blake512_close(&ctx_blake, output);
		break;
	case 2:
		sph_groestl512_context ctx_groestl;
		sph_groestl512_init(&ctx_groestl);
		sph_groestl512(&ctx_groestl, input, 64);
		sph_groestl512_close(&ctx_groestl, output);
		break;
	case 3:
		sph_skein512_context ctx_skein;
		sph_skein512_init(&ctx_skein);
		sph_skein512(&ctx_skein, input, 64);
		sph_skein512_close(&ctx_skein, output);
		break;
	case 4:
		sph_luffa512_context ctx_luffa;
		sph_luffa512_init(&ctx_luffa);
		sph_luffa512(&ctx_luffa, input, 64);
		sph_luffa512_close(&ctx_luffa, output);
		break;
	case 5:
		sph_echo512_context ctx_echo;
		sph_echo512_init(&ctx_echo);
		sph_echo512(&ctx_echo, input, 64);
		sph_echo512_close(&ctx_echo, output);
		break;
	case 6:
		sph_shavite512_context ctx_shavite;
		sph_shavite512_init(&ctx_shavite);
		sph_shavite512(&ctx_shavite, input, 64);
		sph_shavite512_close(&ctx_shavite, output);
		break;
	case 7:
		sph_fugue512_context ctx_fugue;
		sph_fugue512_init(&ctx_fugue);
		sph_fugue512(&ctx_fugue, input, 64);
		sph_fugue512_close(&ctx_fugue, output);
		break;
	case 8:
		sph_simd512_context ctx_simd;
		sph_simd512_init(&ctx_simd);
		sph_simd512(&ctx_simd, input, 64);
		sph_simd512_close(&ctx_simd, output);
		break;
	case 9:
		sph_cubehash512_context ctx_cubehash;
		sph_cubehash512_init(&ctx_cubehash);
		sph_cubehash512(&ctx_cubehash, input, 64);
		sph_cubehash512_close(&ctx_cubehash, output);
		break;
	default:
		break;
	}
}

extern "C" static void shiftr_lp(const uint32_t *input, uint32_t *output, unsigned int shift)
{
	if (!shift) {
		memcpy(output, input, 64);
		return;
	}
	memset(output, 0, 64);
	for (int i = 0; i < 15; ++i) {
		output[i + 1] |= (input[i] >> (32 - shift));
		output[i] |= (input[i] << shift);
	}
	output[15] |= (input[15] << shift);
	return;
}

// CPU HASH
extern "C" void drophash(void *output, const void *input)
{
	sph_jh512_context ctx_jh;
	uchar _ALIGN(64) hash[2][64];
	uint32_t *phashA = (uint32_t *)hash[0];
	uint32_t *phashB = (uint32_t *)hash[1];

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, (const void*)input, 80);
	sph_jh512_close(&ctx_jh, (void*)phashA);

	int startPosition = phashA[0] % 31;

	for (int i = startPosition; i < 31; i--) {
		int start = i % 10;
		for (int j = start; j < 10; j++) {
			shiftr_lp(phashA, phashB, (i & 3));
			switch_hash((const void*)phashB, (void*)phashA, j);
		}
		for (int j = 0; j < start; j++) {
			shiftr_lp(phashA, phashB, (i & 3));
			switch_hash((const void*)phashB, (void*)phashA, j);
		}
		i += 10;
	}
	for (int i = 0; i < startPosition; i--) {
		int start = i % 10;
		for (int j = start; j < 10; j++) {
			shiftr_lp(phashA, phashB, (i & 3));
			switch_hash((const void*)phashB, (void*)phashA, j);
		}
		for (int j = 0; j < start; j++) {
			shiftr_lp(phashA, phashB, (i & 3));
			switch_hash((const void*)phashB, (void*)phashA, j);
		}
		i += 10;
	}
	memcpy(output, phashA, 32);
}

// ------------------------------------------------------------------------------------------------

#ifdef DROP_OFFF
__global__ __launch_bounds__(128, 8)
void drop_get_poks_gpu(uint32_t threads, uint32_t *d_hash, uint16_t *d_poks)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		d_poks[thread] = ((uint16_t*)&d_hash[thread * 16U])[1];
	}
}

__host__
void drop_get_poks(int thr_id, uint32_t threads, uint32_t *d_hash, uint16_t* d_poks)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	drop_get_poks_gpu << <grid, block >> > (threads, d_hash, d_poks);
}

#endif

extern void drop_jh512_cpu_init(int thr_id, uint32_t threads);
extern void drop_jh512_cpu_setBlock80(void *pdata);

extern void drop_jh512_cpu_hash_80a(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order,  uint64_t *d_roundInfo);
extern void drop_jh512_cpu_hash_80b(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order,  uint64_t *d_roundInfo);

extern void quark_keccak512_cpu_init(int thr_id, uint32_t threads);
extern void quark_keccak512_cpu_hash_64_drop(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order,  uint64_t *d_roundInfo, int round, int subRound);

extern void quark_blake512_cpu_init(int thr_id, uint32_t threads);
extern void quark_blake512_cpu_hash_64_drop(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash, int order,  uint64_t *d_roundInfo, int round, int subRound);

extern void quark_groestl512_cpu_init(int thr_id, uint32_t threads);
extern void quark_groestl512_cpu_hash_64_drop(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order,  uint64_t *d_roundInfo, int round, int subRound);

extern void quark_skein512_cpu_init(int thr_id, uint32_t threads);
extern void quark_skein512_cpu_hash_64_drop(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order,  uint64_t *d_roundInfo, int round, int subRound);

extern void x11_luffa512_cpu_init(int thr_id, uint32_t threads);
extern void x11_luffa512_cpu_hash_64_drop(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order,  uint64_t *d_roundInfo, int round, int subRound);

extern void x11_echo512_cpu_init(int thr_id, uint32_t threads);
extern void x11_echo512_cpu_hash_64_drop(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order,  uint64_t *d_roundInfo, int round, int subRound);

extern void x11_shavite512_cpu_init(int thr_id, uint32_t threads);
extern void x11_shavite512_cpu_hash_64_drop(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order,  uint64_t *d_roundInfo, int round, int subRound);

extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64_drop(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order,  uint64_t *d_roundInfo, int round, int subRound);

extern void drop_simd512_cpu_init(int thr_id, uint32_t threads);
extern void drop_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order, uint64_t * d_roundInfo, int round, int subRound);

extern void x11_cubehash512_cpu_init(int thr_id, uint32_t threads);
extern void x11_cubehash512_cpu_hash_64_drop(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order,  uint64_t *d_roundInfo, int round, int subRound);


extern "C" void drophash_pok(void *output, uint32_t *pdata, bool xnonce)
{
	const uint32_t version = pdata[0] & 0x0000FFFF;
	uint32_t _ALIGN(64) hash[8];
	uint32_t pok;

	pdata[0] = version;
	drophash(hash, pdata);

	// fill PoK
	pok = version | (hash[0] & 0xFFFF0000);
	if (pdata[0] != pok) {
		pdata[0] = pok;
		drophash(hash, pdata);
	}
	if (xnonce)	pdata[22] = pok;
	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_drop(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	uint32_t max_nonce, unsigned long *hashes_done)
{
#ifdef DROP_OFFF
	uint32_t _ALIGN(64) tmpdata[20];
#else
	const uint32_t oP = pdata[22];
#endif
	const uint32_t first_nonce = pdata[19];

	uint32_t throughpt = device_intensity(thr_id, __func__, 1U << 19);
	const uint32_t throughput = min(throughpt, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		//cudaDeviceReset();
		//cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

		cudaMalloc(&d_hash[thr_id], throughput * 16 * sizeof(uint32_t));
		cudaMalloc(&d_roundInfo[thr_id], throughput * sizeof(uint64_t));
#ifdef DROP_OFFF
		cudaMalloc(&d_poks[thr_id], throughput * sizeof(uint16_t));
#endif
		CUDA_SAFE_CALL(cudaGetLastError());
		drop_jh512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_blake512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput);
		x11_echo512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		drop_simd512_cpu_init(thr_id, throughput);
		x11_cubehash512_cpu_init(thr_id, throughput);

		cuda_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	drop_jh512_cpu_setBlock80((void*)pdata);
	cuda_check_cpu_setTarget(ptarget);

#ifdef DR_TIMINGS
	UINT timePeriod = 1;
	timeBeginPeriod(timePeriod);
	long prTime;
#endif
	do {
#ifdef DR_TIMINGS
		prTime = timeGetTime();
#endif
		int order = 0;
		// Hash with CUDA - round 1 of 2
		drop_jh512_cpu_hash_80a(thr_id, throughput, pdata[19], d_hash[thr_id], order++,d_roundInfo[thr_id]);
		//cudaDeviceSynchronize();

		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 10; j++) {
				quark_keccak512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				quark_blake512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				quark_groestl512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				quark_skein512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				x11_luffa512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				x11_echo512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				x11_shavite512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				x13_fugue512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				drop_simd512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				x11_cubehash512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				//cudaDeviceSynchronize();
			}
		}
#ifdef DROP_OFFF
		drop_get_poks(thr_id, throughput, d_hash[thr_id], d_poks[thr_id]);
#endif
		// Hash with CUDA - round 2 of 2
		drop_jh512_cpu_hash_80b(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id]);
		//cudaDeviceSynchronize();

		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 10; j++) {
				quark_keccak512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				quark_blake512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				quark_groestl512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				quark_skein512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				x11_luffa512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				x11_echo512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				x11_shavite512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				x13_fugue512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				drop_simd512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], order++, d_roundInfo[thr_id], i, j);
				x11_cubehash512_cpu_hash_64_drop(thr_id, throughput, pdata[19], d_hash[thr_id], order++,d_roundInfo[thr_id], i, j);
				//cudaDeviceSynchronize();
			}
		}

#ifdef DR_TIMINGS
		prTime = timeGetTime() - prTime;
		printf("\nTotal time elapsed: %f seconds\n", (prTime / 1000.0));
		timeEndPeriod(timePeriod);
#endif

		uint32_t foundNonce = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (foundNonce != UINT32_MAX)
		{
			uint32_t vhash64[8];
			uint32_t oldp0 = pdata[0];
			uint32_t oldp19 = pdata[19];
#ifdef DROP_OFFF
			const uint32_t version = pdata[0] & 0x0000FFFF;
			memcpy(tmpdata, pdata, 80);

			uint32_t offset = foundNonce - pdata[19];
			uint32_t pok = 0;
			uint16_t h_pok;

			*hashes_done = pdata[19] - first_nonce + throughput;

			cudaMemcpy(&h_pok, d_poks[thr_id] + offset, sizeof(uint16_t), cudaMemcpyDeviceToHost);
			pok = version | (0x10000UL * h_pok);
			pdata[0] = pok; pdata[19] = foundNonce;
			drophash(vhash64, pdata);
			if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
				int res = 1;
				uint32_t secNonce = cuda_check_hash_suppl(thr_id, throughput, oldp19, d_hash[thr_id], 1);
				if (secNonce != 0) {
					offset = secNonce - oldp19;
					cudaMemcpy(&h_pok, d_poks[thr_id] + offset, sizeof(uint16_t), cudaMemcpyDeviceToHost);
					pok = version | (0x10000UL * h_pok);
					memcpy(tmpdata, pdata, 80);
					tmpdata[0] = pok; tmpdata[19] = secNonce;
					drophash(vhash64, tmpdata);
					if (vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget)) {
						pdata[21] = secNonce;
						pdata[22] = pok;
						res++;
					}
				}
#else

			const uint32_t Htarg = ptarget[7];

			*hashes_done = pdata[19] - first_nonce + throughput;

			pdata[19] = foundNonce;
			drophash_pok(vhash64, pdata, false);

			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;
				uint32_t secNonce = cuda_check_hash_suppl(thr_id, throughput, oldp19, d_hash[thr_id], 1);
				if (secNonce != 0)
				{
					pdata[19] = secNonce;
					drophash_pok(vhash64, pdata, true);
					pdata[19] = foundNonce;
					if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
					{
						pdata[21] = secNonce;
						res++;
					}
					else
						pdata[22] = oP;
				}
#endif
				return res;
			}
			else {
				applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", device_map[thr_id], foundNonce);

				pdata[19]++;
				pdata[0] = oldp0;
			}
		}
		else
			pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
