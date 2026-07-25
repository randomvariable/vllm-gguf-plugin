// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/convert.cu
// Dequant functions
static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = __int2half_rn(vui & 0xF);
    v.y = __int2half_rn(vui >> 4);

    v = __hsub2(v, __floats2half2_rn(8.0f, 8.0f));
    v = __hmul2(v, {d, d});
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = __int2half_rn(vui & 0xF);
    v.y = __int2half_rn(vui >> 4);

    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = __int2half_rn((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = __int2half_rn((x[ib].qs[iqs] >>  4) | xh_1);

    v = __hsub2(v, __floats2half2_rn(16.0f, 16.0f));
    v = __hmul2(v, {d, d});
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = __int2half_rn((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = __int2half_rn((x[ib].qs[iqs] >>  4) | xh_1);

    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x = __int2half_rn(x[ib].qs[iqs + 0]);
    v.y = __int2half_rn(x[ib].qs[iqs + 1]);

    v = __hmul2(v, {d, d});
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k) {
    const int64_t i = 2*((int64_t)blockDim.x*blockIdx.x + threadIdx.x);

    if (i >= k) {
        return;
    }

    const int ib = i/qk; // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0]        = convert_from_half<dst_t>(v.x);
    y[iybs + iqs + y_offset] = convert_from_half<dst_t>(v.y);
}

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const auto tid = threadIdx.x;
    const int n   = tid/32;
    const int l   = tid - 32*n;
    const int is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    half dall = __low2half(x[i].dm);
    half dmin = __high2half(x[i].dm);
    y[l+ 0] = convert_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is+0] & 0xF) * ((q >> 0) & 3))), __hmul(dmin,  __int2half_rn(x[i].scales[is+0] >> 4))));
    y[l+32] = convert_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is+2] & 0xF) * ((q >> 2) & 3))), __hmul(dmin,  __int2half_rn(x[i].scales[is+2] >> 4))));
    y[l+64] = convert_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is+4] & 0xF) * ((q >> 4) & 3))), __hmul(dmin,  __int2half_rn(x[i].scales[is+4] >> 4))));
    y[l+96] = convert_from_half<dst_t>(__hsub(__hmul(dall, __int2half_rn((x[i].scales[is+6] & 0xF) * ((q >> 6) & 3))), __hmul(dmin,  __int2half_rn(x[i].scales[is+6] >> 4))));
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *) vx;

    const auto r = threadIdx.x/4;
    const int tid = r/2;
    const int is0 = r%2;
    const int l0 = 16*is0 + 4*(threadIdx.x%4);
    const int n = tid / 4;
    const int j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    half d_all = x[i].d;
    half dl = __hmul(d_all,  __int2half_rn(us - 32));

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) {
        y[l] = convert_from_half<dst_t>(__hmul(dl,  __int2half_rn((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4))));
    }
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const auto i = blockIdx.x;

    // assume 32 threads
    const auto tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int is  = 2*il;
    const int n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const half dall = __low2half(x[i].dm);
    const half dmin = __high2half(x[i].dm);

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const half d1 = __hmul(dall, __int2half_rn(sc));
    const half m1 = __hmul(dmin,  __int2half_rn(m));
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const half d2 = __hmul(dall, __int2half_rn(sc));
    const half m2 = __hmul(dmin, __int2half_rn(m));
    for (int l = 0; l < n; ++l) {
        y[l + 0] = convert_from_half<dst_t>(__hsub(__hmul(d1, __int2half_rn(q[l] & 0xF)), m1));
        y[l +32] = convert_from_half<dst_t>(__hsub(__hmul(d2,  __int2half_rn(q[l] >> 4)), m2));
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const auto i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const auto tid = threadIdx.x;
    const int il  = tid/16;   // il is in 0...3
    const int ir  = tid%16;   // ir is in 0...15
    const int is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const half dall = __low2half(x[i].dm);
    const half dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const half d1 = __hmul(dall, __int2half_rn(sc)); const half m1 = __hmul(dmin, __int2half_rn(m));
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const half d2 = __hmul(dall, __int2half_rn(sc)); const half m2 = __hmul(dmin, __int2half_rn(m));

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = convert_from_half<dst_t>(__hsub(__hmul(d1, __int2half_rn((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0))), m1));
    y[ 1] = convert_from_half<dst_t>(__hsub(__hmul(d1, __int2half_rn((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0))), m1));
    hm <<= 1;
    y[32] = convert_from_half<dst_t>(__hsub(__hmul(d2, __int2half_rn((ql[0] >>  4) + (qh[0] & hm ? 16 : 0))), m2));
    y[33] = convert_from_half<dst_t>(__hsub(__hmul(d2, __int2half_rn((ql[1] >>  4) + (qh[1] & hm ? 16 : 0))), m2));
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const auto i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const auto tid = threadIdx.x;
    const int ip  = tid/32;   // ip is 0 or 1
    const int il  = tid - 32*ip; // 0...32
    const int is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const half d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = convert_from_half<dst_t>(__hmul(d, __int2half_rn(sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32))));
    y[32] = convert_from_half<dst_t>(__hmul(d, __int2half_rn(sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32))));
    y[64] = convert_from_half<dst_t>(__hmul(d, __int2half_rn(sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32))));
    y[96] = convert_from_half<dst_t>(__hmul(d, __int2half_rn(sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32))));
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i   = blockIdx.x;
    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const auto tid = threadIdx.x;
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = __half2float(x[i].d) * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i   = blockIdx.x;
    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const auto tid = threadIdx.x;
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = __half2float(x[i].d) * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);

}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i   = blockIdx.x;
    const block_iq2_s * x = (const block_iq2_s *) vx;

    const auto tid = threadIdx.x;
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    const float d = __half2float(x[i].d) * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i   = blockIdx.x;
    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const auto tid = threadIdx.x;
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t  * q3 = x[i].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = __half2float(x[i].d) * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i   = blockIdx.x;
    const block_iq3_s * x = (const block_iq3_s *) vx;

    const auto tid = threadIdx.x;
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * qs = x[i].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3xs_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3xs_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    const float d = __half2float(x[i].d) * (0.5f + ((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf)) * 0.5f;
    const uint8_t signs = x[i].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const float delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = __half2float(x[i].d) * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = __half2float(scale.f16) * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i   = blockIdx.x;
    const block_iq4_nl * x = (const block_iq4_nl *) vx + i*(QK_K/QK4_NL);

    const auto tid = threadIdx.x;
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = __half2float(x[ib].d);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }

}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const auto i   = blockIdx.x;
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const auto tid = threadIdx.x;
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = __half2float(x[i].d) * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int64_t num_blocks = (k + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE);
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

// =============================================================================
// ROCmFPX shared UE4M3 -> FP32 scale decoder
// Ported from rocmfp4_hip_scale.cuh:111-125 (rocmfpx_ue4m3_to_fp32_finite)
// UE4M3: unsigned E4M3 (4-bit exponent, 3-bit mantissa, no sign)
// Valid range: 0x00..0x7E; values > 0x7E decode to 0.
// Subnormal (exp==0): man * 2^-10 = man * (1/1024)
// Normal: fp32 bits = (exp + 119) << 23 | (man << 20), i.e. bias=8 not 15
// =============================================================================
__device__ __forceinline__ float rocmfpx_decode_scale_to_fp32(uint8_t x) {
    if (x > 0x7e) {
        return 0.0f;
    }

    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;

    if (exp == 0) {
        return (float) man * (1.0f / 1024.0f);
    }

    const uint32_t bits =
        ((uint32_t) exp + 119u) << 23 | ((uint32_t) man << 20);
    return __uint_as_float(bits);
}

// Backward-compatible alias for existing Q4 code
#define rocmfp4_decode_scale_to_fp32 rocmfpx_decode_scale_to_fp32

// =============================================================================
// ROCmFPX Q4_0 dequantization (GGML type 100)
// Ported from ROCmFPX/ggml/rocmfp4/rocmfp4.c
// =============================================================================

// Codebook10: maps 4-bit index to signed int8 value
// Indices 0-7: {0, 1, 2, 3, 4, 6, 8, 10}
// Indices 8-15: {0, -1, -2, -3, -4, -6, -8, -10}
__constant__ static const int8_t ROCMFP4_CODEBOOK10[16] = {
    0, 1, 2, 3, 4, 6, 8, 10,
    0, -1, -2, -3, -4, -6, -8, -10
};

// =============================================================================
// ROCmFPX Q2_0 dequantization (GGML type 107, aka iFP2)
// Ported from ROCmFPX/ggml/rocmfpx/rocmfpx.c:240-244 (S40 FP2 codebook)
// Codebook S40 MORD: {-4, -1, +1, +4} (2-bit index -> signed value)
// =============================================================================
__constant__ static const int8_t ROCMFP2_CODEBOOK_S40[4] = {
    -4, -1, 1, 4
};

// =============================================================================
// ROCmFPX Q3_0 dequantization (GGML type 104)
// Ported from ROCmFPX/ggml/rocmfpx/rocmfpx.c (FP3 codebook)
// Codebook: {0, 1, 2, 4, 0, -1, -2, -4} (3-bit index, 8 entries)
// =============================================================================
__constant__ static const int8_t ROCMFP3_CODEBOOK[8] = {
    0, 1, 2, 4, 0, -1, -2, -4
};

// Global kernel for Q4_0_ROCMFP4 dequantization
// Each thread handles one GGML block (32 weights from 18 bytes)
__global__ void dequantize_block_rocmfp4_q4_0(const void* __restrict__ vx,
                                               dst_t* __restrict__ y,
                                               int64_t k) {
    const int64_t nb = (k + QK4_0_ROCMFP4 - 1) / QK4_0_ROCMFP4;
    const block_rocmfp4* __restrict__ x =
        static_cast<const block_rocmfp4*>(vx);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nb;
         i += blockDim.x * gridDim.x) {
        float d0 = rocmfp4_decode_scale_to_fp32(x[i].e[0]);
        float d1 = rocmfp4_decode_scale_to_fp32(x[i].e[1]);

        for (int j = 0; j < 16; j++) {
            uint8_t q = x[i].qs[j];
            int8_t v_lo = ROCMFP4_CODEBOOK10[q & 0x0f];
            int8_t v_hi = ROCMFP4_CODEBOOK10[q >> 4];
            y[i * QK4_0_ROCMFP4 + j * 2 + 0] =
                static_cast<dst_t>(v_lo) * d0;
            y[i * QK4_0_ROCMFP4 + j * 2 + 1] =
                static_cast<dst_t>(v_hi) * d1;
        }
    }
}

template<typename dst_t>
static void dequantize_row_rocmfp4_q4_0_cuda(const void * vx, dst_t * y,
                                              const int64_t k,
                                              cudaStream_t stream) {
    const int nb = (k + QK4_0_ROCMFP4 - 1) / QK4_0_ROCMFP4;
    const int nblocks = (nb + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) /
                        CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block_rocmfp4_q4_0<<<nblocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0,
                                     stream>>>(vx, y, k);
}

// =============================================================================
// ROCmFPX Q4_0_FAST dequantization (GGML type 101)
// Fast variant: 1 UE4M3 scale shared across all 32 weights (17 bytes/block)
// Same Codebook10 as the standard Q4 variant
// =============================================================================
__global__ void dequantize_block_rocmfp4_fast_q4_0(const void* __restrict__ vx,
                                                    dst_t* __restrict__ y,
                                                    int64_t k) {
    const int64_t nb = (k + QK4_0_ROCMFP4 - 1) / QK4_0_ROCMFP4;
    const block_rocmfp4_fast* __restrict__ x =
        static_cast<const block_rocmfp4_fast*>(vx);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nb;
         i += blockDim.x * gridDim.x) {
        float d = rocmfpx_decode_scale_to_fp32(x[i].e[0]);

        for (int j = 0; j < 16; j++) {
            uint8_t q = x[i].qs[j];
            int8_t v_lo = ROCMFP4_CODEBOOK10[q & 0x0f];
            int8_t v_hi = ROCMFP4_CODEBOOK10[q >> 4];
            y[i * QK4_0_ROCMFP4 + j * 2 + 0] =
                static_cast<dst_t>(v_lo) * d;
            y[i * QK4_0_ROCMFP4 + j * 2 + 1] =
                static_cast<dst_t>(v_hi) * d;
        }
    }
}

template<typename dst_t>
static void dequantize_row_rocmfp4_fast_q4_0_cuda(const void * vx, dst_t * y,
                                                   const int64_t k,
                                                   cudaStream_t stream) {
    const int nb = (k + QK4_0_ROCMFP4 - 1) / QK4_0_ROCMFP4;
    const int nblocks = (nb + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) /
                        CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block_rocmfp4_fast_q4_0<<<nblocks, CUDA_DEQUANTIZE_BLOCK_SIZE,
                                          0, stream>>>(vx, y, k);
}

// =============================================================================
// ROCmFPX Q2_0 dequantization (GGML type 107, aka iFP2)
// Ported from ROCmFPX/ggml/rocmfpx/rocmfpx.c:381-396 (rocmfpx_dequantize_row_fp2)
// Each thread handles one GGML block (32 weights from 10 bytes)
// =============================================================================
__global__ void dequantize_block_rocmfpx_q2_0(const void* __restrict__ vx,
                                               dst_t* __restrict__ y,
                                               int64_t k) {
    const int64_t nb = (k + QK2_0_ROCMFPX - 1) / QK2_0_ROCMFPX;
    const block_rocmfp2* __restrict__ x =
        static_cast<const block_rocmfp2*>(vx);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nb;
         i += blockDim.x * gridDim.x) {
        float d0 = rocmfpx_decode_scale_to_fp32(x[i].e[0]);
        float d1 = rocmfpx_decode_scale_to_fp32(x[i].e[1]);

        for (int half = 0; half < 2; half++) {
            float scale = (half == 0) ? d0 : d1;
            for (int j = 0; j < 16; j++) {
                uint8_t code = (x[i].qs[half * 4 + j / 4] >> (2 * (j % 4))) & 3u;
                int8_t val = ROCMFP2_CODEBOOK_S40[code];
                y[i * QK2_0_ROCMFPX + half * 16 + j] =
                    static_cast<dst_t>(val) * scale;
            }
        }
    }
}

template<typename dst_t>
static void dequantize_row_rocmfpx_q2_0_cuda(const void * vx, dst_t * y,
                                              const int64_t k,
                                              cudaStream_t stream) {
    const int nb = (k + QK2_0_ROCMFPX - 1) / QK2_0_ROCMFPX;
    const int nblocks = (nb + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) /
                        CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block_rocmfpx_q2_0<<<nblocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0,
                                     stream>>>(vx, y, k);
}

// =============================================================================
// ROCmFPX Q3_0 dequantization (GGML type 104)
// Ported from ROCmFPX/ggml/rocmfpx/rocmfpx.c (rocmfpx_dequantize_row_fp3)
// 3-bit indices packed 8 per 3 bytes
// Each thread handles one GGML block (32 weights from 14 bytes)
// =============================================================================
__global__ void dequantize_block_rocmfpx_q3_0(const void* __restrict__ vx,
                                               dst_t* __restrict__ y,
                                               int64_t k) {
    const int64_t nb = (k + QK3_0_ROCMFPX - 1) / QK3_0_ROCMFPX;
    const block_rocmfp3* __restrict__ x =
        static_cast<const block_rocmfp3*>(vx);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nb;
         i += blockDim.x * gridDim.x) {
        float d0 = rocmfpx_decode_scale_to_fp32(x[i].e[0]);
        float d1 = rocmfpx_decode_scale_to_fp32(x[i].e[1]);

        // Unpack 32 x 3-bit codes from 12 bytes
        // Each group of 8 codes is packed in 3 bytes
        uint8_t codes[32];
        for (int g = 0; g < 4; g++) {
            const uint8_t* src = x[i].qs + g * 3;
            uint8_t* dst = codes + g * 8;
            dst[0] = src[0] & 0x07;
            dst[1] = (src[0] >> 3) & 0x07;
            dst[2] = ((src[0] >> 6) | (src[1] << 2)) & 0x07;
            dst[3] = (src[1] >> 1) & 0x07;
            dst[4] = (src[1] >> 4) & 0x07;
            dst[5] = ((src[1] >> 7) | (src[2] << 1)) & 0x07;
            dst[6] = (src[2] >> 2) & 0x07;
            dst[7] = (src[2] >> 5) & 0x07;
        }

        for (int half = 0; half < 2; half++) {
            float scale = (half == 0) ? d0 : d1;
            for (int j = 0; j < 16; j++) {
                uint8_t code = codes[half * 16 + j];
                int8_t val = ROCMFP3_CODEBOOK[code & 0x07];
                y[i * QK3_0_ROCMFPX + half * 16 + j] =
                    static_cast<dst_t>(val) * scale;
            }
        }
    }
}

template<typename dst_t>
static void dequantize_row_rocmfpx_q3_0_cuda(const void * vx, dst_t * y,
                                              const int64_t k,
                                              cudaStream_t stream) {
    const int nb = (k + QK3_0_ROCMFPX - 1) / QK3_0_ROCMFPX;
    const int nblocks = (nb + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) /
                        CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block_rocmfpx_q3_0<<<nblocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0,
                                     stream>>>(vx, y, k);
}

// =============================================================================
// ROCmFPX Q6_0 dequantization (GGML type 102)
// Ported from ROCmFPX/ggml/rocmfpx/rocmfpx.c:1160-1185 (rocmfpx_dequantize_row_fp6)
// 6-bit sign-magnitude codes packed 4 per 3 bytes
// Each thread handles one GGML block (32 weights from 26 bytes)
// =============================================================================
__device__ __forceinline__ int rocmfpx_decode_fp6_code(uint8_t code) {
    // 6-bit sign-magnitude: bit 5 = sign, bits 0-4 = magnitude (0-31)
    // Asymmetric range [-32, +31]: code 0x20 (sign=1, mag=0) -> -32
    // Ported from rocmfpx.c:677-679
    const int mag = code & 31u;
    return (code & 32u) ? -(mag == 0 ? 32 : mag) : mag;
}

__device__ __forceinline__ void rocmfpx_fp6_unpack4(const uint8_t* src,
                                                     uint8_t* dst) {
    // Unpack 4 x 6-bit codes from 3 bytes
    dst[0] = src[0] & 0x3F;
    dst[1] = ((src[0] >> 6) | (src[1] << 2)) & 0x3F;
    dst[2] = ((src[1] >> 4) | (src[2] << 4)) & 0x3F;
    dst[3] = (src[2] >> 2) & 0x3F;
}

__global__ void dequantize_block_rocmfpx_q6_0(const void* __restrict__ vx,
                                               dst_t* __restrict__ y,
                                               int64_t k) {
    const int64_t nb = (k + QK6_0_ROCMFPX - 1) / QK6_0_ROCMFPX;
    const block_rocmfp6* __restrict__ x =
        static_cast<const block_rocmfp6*>(vx);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nb;
         i += blockDim.x * gridDim.x) {
        float d0 = rocmfpx_decode_scale_to_fp32(x[i].e[0]);
        float d1 = rocmfpx_decode_scale_to_fp32(x[i].e[1]);

        // Unpack all 32 codes in 8 groups of 4 (8 x 3 bytes = 24 bytes)
        uint8_t codes[32];
        rocmfpx_fp6_unpack4(x[i].qs,      codes);
        rocmfpx_fp6_unpack4(x[i].qs +  3, codes +  4);
        rocmfpx_fp6_unpack4(x[i].qs +  6, codes +  8);
        rocmfpx_fp6_unpack4(x[i].qs +  9, codes + 12);
        rocmfpx_fp6_unpack4(x[i].qs + 12, codes + 16);
        rocmfpx_fp6_unpack4(x[i].qs + 15, codes + 20);
        rocmfpx_fp6_unpack4(x[i].qs + 18, codes + 24);
        rocmfpx_fp6_unpack4(x[i].qs + 21, codes + 28);

        for (int half = 0; half < 2; half++) {
            float scale = (half == 0) ? d0 : d1;
            for (int j = 0; j < 16; j++) {
                int idx = half * 16 + j;
                int val = rocmfpx_decode_fp6_code(codes[idx]);
                y[i * QK6_0_ROCMFPX + idx] =
                    static_cast<dst_t>(val) * scale;
            }
        }
    }
}

template<typename dst_t>
static void dequantize_row_rocmfpx_q6_0_cuda(const void * vx, dst_t * y,
                                              const int64_t k,
                                              cudaStream_t stream) {
    const int nb = (k + QK6_0_ROCMFPX - 1) / QK6_0_ROCMFPX;
    const int nblocks = (nb + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) /
                        CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block_rocmfpx_q6_0<<<nblocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0,
                                     stream>>>(vx, y, k);
}

// =============================================================================
// ROCmFPX Q8_0 dequantization (GGML type 103)
// Ported from ROCmFPX/ggml/rocmfpx/rocmfpx.c (rocmfpx_dequantize_row_fp8)
// Direct signed int8 clamped [-127, 127]
// Each thread handles one GGML block (32 weights from 33 bytes)
// =============================================================================
__global__ void dequantize_block_rocmfpx_q8_0(const void* __restrict__ vx,
                                               dst_t* __restrict__ y,
                                               int64_t k) {
    const int64_t nb = (k + QK8_0_ROCMFPX - 1) / QK8_0_ROCMFPX;
    const block_rocmfp8* __restrict__ x =
        static_cast<const block_rocmfp8*>(vx);

    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nb;
         i += blockDim.x * gridDim.x) {
        float scale = rocmfpx_decode_scale_to_fp32(x[i].e[0]);

        for (int j = 0; j < 32; j++) {
            int val = (int) x[i].qs[j];
            y[i * QK8_0_ROCMFPX + j] = static_cast<dst_t>(val) * scale;
        }
    }
}

template<typename dst_t>
static void dequantize_row_rocmfpx_q8_0_cuda(const void * vx, dst_t * y,
                                              const int64_t k,
                                              cudaStream_t stream) {
    const int nb = (k + QK8_0_ROCMFPX - 1) / QK8_0_ROCMFPX;
    const int nblocks = (nb + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) /
                        CUDA_DEQUANTIZE_BLOCK_SIZE;
    dequantize_block_rocmfpx_q8_0<<<nblocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0,
                                     stream>>>(vx, y, k);
}

// =============================================================================
// ik_llama.cpp K-variant i-quant dequant kernels
// Ported from ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp
// All use QK_K=256 super-blocks, software-dequant, dual-codebook lookup.
// =============================================================================

// Codebook tables for K-variant i-quant formats
__device__ __constant__ int8_t kvalues_iq2nl[8] = {
    -31, -13, 1, 17,   -26, -8, 6, 22
};

__device__ __constant__ int8_t kvalues_iq3nl[16] = {
    -63, -40, -23, -10, 1, 13, 28,  47,
    -59, -36, -19,  -6, 5, 17, 32,  51,
};

// iq4k_values already exists as kvalues_iq4nl[16] above; we need the full
// 32-entry table for IQ4_K/IQ4_KS.
__device__ __constant__ int8_t kvalues_iq4k[32] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117,
};

// IQ2_K: 78 bytes/256w, 2-bit dual-codebook
// Block: half d, uint16 extra, uint8 scales[8], uint8 qs[64]
// Per ib32 (32 weights): nibble scale, dual codebook via extra bits
__global__ void dequantize_block_iq2_k(const void* __restrict__ vx,
                                        dst_t* __restrict__ yy) {
    const int i = blockIdx.x;
    const block_iq2_k* __restrict__ x = (const block_iq2_k*)vx;
    const int tid = threadIdx.x;
    const int ib32 = tid;  // 0..7, each thread handles one 32-weight group

    if (ib32 >= QK_K / 32) return;

    const float d = __half2float(x[i].d);
    uint16_t extra = x[i].extra;
    const uint8_t* qs = x[i].qs;
    const uint8_t* scales = x[i].scales;

    // Each thread processes its own ib32 group
    // shift = 2 * (ib32 % 4), qs advances by 32 bytes every 4 ib32s
    int shift = 2 * (ib32 % 4);
    const uint8_t* qs_base = qs + (ib32 / 4) * 32;

    float dl1 = d * ((scales[ib32] & 0xf) - 8);
    float dl2 = d * ((scales[ib32] >> 4) - 8);

    // extra bits: bit 0 selects codebook for dl1, bit 1 for dl2
    // But extra is per-super-block and bits are consumed in ib32 order.
    // We need the original extra bits for THIS ib32.
    uint16_t extra_shifted = extra >> (2 * ib32);
    const int8_t* values1 = (extra_shifted & 1) ? kvalues_iq2nl + 4 : kvalues_iq2nl;
    const int8_t* values2 = (extra_shifted & 2) ? kvalues_iq2nl + 4 : kvalues_iq2nl;

    dst_t* y = yy + i * QK_K + ib32 * 32;

    for (int j = 0; j < 16; ++j) {
        y[j + 0]  = dl1 * values1[(qs_base[j + 0]  >> shift) & 3];
        y[j + 16] = dl2 * values2[(qs_base[j + 16] >> shift) & 3];
    }
}

template<typename dst_t>
static void dequantize_row_iq2_k_cuda(const void* vx, dst_t* y,
                                       const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq2_k<<<nb, 32, 0, stream>>>(vx, y);
}

// IQ3_K: 110 bytes/256w, 3-bit dual-codebook with signed scales
// Block: half d, uint16 extra, uint16 scales_h, uint8 scales_l[8],
//        uint8 qs[64], uint8 qh[32]
__global__ void dequantize_block_iq3_k(const void* __restrict__ vx,
                                        dst_t* __restrict__ yy) {
    const int i = blockIdx.x;
    const block_iq3_k* __restrict__ x = (const block_iq3_k*)vx;
    const int tid = threadIdx.x;
    const int ib32 = tid;  // 0..7

    if (ib32 >= QK_K / 32) return;

    const float d = __half2float(x[i].d);
    uint16_t sh = x[i].scales_h;
    uint16_t extra = x[i].extra;
    const uint8_t* qs = x[i].qs;
    const uint8_t* qh = x[i].qh;
    const uint8_t* scales_l = x[i].scales_l;

    // Scale: 2*(nibble)+1 with sign from scales_h
    // scales_h provides 2 sign bits per ib32
    uint16_t sh_local = sh >> (2 * ib32);
    float dl1 = d * ((2 * (scales_l[ib32] & 0xf) + 1) * ((sh_local & 1) ? -1 : 1));
    float dl2 = d * ((2 * (scales_l[ib32] >> 4) + 1) * ((sh_local & 2) ? -1 : 1));

    // Codebook selection via extra bits
    uint16_t extra_shifted = extra >> (2 * ib32);
    const int8_t* values1 = (extra_shifted & 1) ? kvalues_iq3nl + 8 : kvalues_iq3nl;
    const int8_t* values2 = (extra_shifted & 2) ? kvalues_iq3nl + 8 : kvalues_iq3nl;

    // 3-bit code: low 2 bits from qs, high bit from qh
    int shift_l = 2 * (ib32 % 4);
    int shift_h = ib32 % 8;
    const uint8_t* qs_base = qs + (ib32 / 4) * 32;

    dst_t* y = yy + i * QK_K + ib32 * 32;

    for (int j = 0; j < 16; ++j) {
        int code1 = ((qs_base[j + 0]  >> shift_l) & 3) | (((qh[j + 0]  >> shift_h) & 1) << 2);
        int code2 = ((qs_base[j + 16] >> shift_l) & 3) | (((qh[j + 16] >> shift_h) & 1) << 2);
        y[j + 0]  = dl1 * values1[code1];
        y[j + 16] = dl2 * values2[code2];
    }
}

template<typename dst_t>
static void dequantize_row_iq3_k_cuda(const void* vx, dst_t* y,
                                       const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq3_k<<<nb, 32, 0, stream>>>(vx, y);
}

// IQ4_K: 150 bytes/256w, 4-bit dual-codebook with 6-bit signed scales
// Block: half d, uint16 extra, uint8 scales_h[4], uint8 scales_l[8],
//        uint8 qs[128]
__global__ void dequantize_block_iq4_k(const void* __restrict__ vx,
                                        dst_t* __restrict__ yy) {
    const int i = blockIdx.x;
    const block_iq4_k* __restrict__ x = (const block_iq4_k*)vx;
    const int tid = threadIdx.x;
    const int ib32 = tid;  // 0..7

    if (ib32 >= QK_K / 32) return;

    const float d = __half2float(x[i].d);
    uint16_t extra = x[i].extra;
    const uint8_t* qs = x[i].qs;

    // 6-bit scale from scales_h and scales_l
    uint8_t sh = x[i].scales_h[ib32 / 2] >> (4 * (ib32 % 2));
    float dl1 = d * (((x[i].scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32);
    float dl2 = d * (((x[i].scales_l[ib32] >> 4) | ((sh << 2) & 0x30)) - 32);

    // Codebook selection via extra bits
    uint16_t extra_shifted = extra >> (2 * ib32);
    const int8_t* values1 = (extra_shifted & 1) ? kvalues_iq4k + 16 : kvalues_iq4k;
    const int8_t* values2 = (extra_shifted & 2) ? kvalues_iq4k + 16 : kvalues_iq4k;

    const uint8_t* qs_base = qs + ib32 * 16;
    dst_t* y = yy + i * QK_K + ib32 * 32;

    for (int j = 0; j < 16; ++j) {
        y[j + 0]  = dl1 * values1[qs_base[j] & 0xf];
        y[j + 16] = dl2 * values2[qs_base[j] >> 4];
    }
}

template<typename dst_t>
static void dequantize_row_iq4_k_cuda(const void* vx, dst_t* y,
                                       const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_k<<<nb, 32, 0, stream>>>(vx, y);
}

// IQ4_KS: 144 bytes/256w + 4-byte row-prefix FP32 scale
// Simplest format: single codebook offset, byte scale per 32-weight group
// Block struct (after FP32 prefix): uint8 scales[8], uint8 qs[128]
// Note: the FP32 prefix is per-ROW, not per-super-block. The block_iq4_ks
// array starts at (float*)x + 1.
__global__ void dequantize_block_iq4_ks(const void* __restrict__ vx,
                                         dst_t* __restrict__ yy,
                                         int64_t k) {
    // The row data starts with a float d prefix before the block array
    const float* dptr = (const float*)vx;
    const float d = *dptr;
    const block_iq4_ks* __restrict__ x = (const block_iq4_ks*)(dptr + 1);

    const int nblock = k / QK_K;
    const int i = blockIdx.x;
    const int ib32 = threadIdx.x;  // 0..7

    if (i >= nblock || ib32 >= QK_K / 32) return;

    const uint8_t* qs = x[i].qs + ib32 * 16;
    uint8_t scale_byte = x[i].scales[ib32];
    float dl = d * ((int)(scale_byte & 254) - 127);
    const int8_t* values = kvalues_iq4k + ((scale_byte & 1) << 4);

    dst_t* y = yy + i * QK_K + ib32 * 32;

    for (int j = 0; j < 16; ++j) {
        y[j + 0]  = dl * values[qs[j] & 0xf];
        y[j + 16] = dl * values[qs[j] >> 4];
    }
}

template<typename dst_t>
static void dequantize_row_iq4_ks_cuda(const void* vx, dst_t* y,
                                        const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_ks<<<nb, 32, 0, stream>>>(vx, y, k);
}

template<typename dst_t>
static to_cuda_ggml_t<dst_t> ggml_get_to_cuda(int64_t type) {
    switch (type) {
        case 2:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case 3:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case 6:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case 7:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case 8:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case 10:
            return dequantize_row_q2_K_cuda;
        case 11:
            return dequantize_row_q3_K_cuda;
        case 12:
            return dequantize_row_q4_K_cuda;
        case 13:
            return dequantize_row_q5_K_cuda;
        case 14:
            return dequantize_row_q6_K_cuda;
        case 16:
            return dequantize_row_iq2_xxs_cuda;
        case 17:
            return dequantize_row_iq2_xs_cuda;
        case 18:
            return dequantize_row_iq3_xxs_cuda;
        case 19:
            return dequantize_row_iq1_s_cuda;
        case 20:
            return dequantize_row_iq4_nl_cuda;
        case 21:
            return dequantize_row_iq3_s_cuda;
        case 22:
            return dequantize_row_iq2_s_cuda;
        case 23:
            return dequantize_row_iq4_xs_cuda;
        case 29:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_Q4_0_ROCMFP4:
            return dequantize_row_rocmfp4_q4_0_cuda;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
            return dequantize_row_rocmfp4_fast_q4_0_cuda;
        case GGML_TYPE_Q2_0_ROCMFPX:
            return dequantize_row_rocmfpx_q2_0_cuda;
        case GGML_TYPE_Q3_0_ROCMFPX:
            return dequantize_row_rocmfpx_q3_0_cuda;
        case GGML_TYPE_Q6_0_ROCMFPX:
            return dequantize_row_rocmfpx_q6_0_cuda;
        case GGML_TYPE_Q8_0_ROCMFPX:
            return dequantize_row_rocmfpx_q8_0_cuda;
        case GGML_TYPE_IQ2_K:
            return dequantize_row_iq2_k_cuda;
        case GGML_TYPE_IQ3_K:
            return dequantize_row_iq3_k_cuda;
        case GGML_TYPE_IQ4_K:
            return dequantize_row_iq4_k_cuda;
        case GGML_TYPE_IQ4_KS:
            return dequantize_row_iq4_ks_cuda;
        default:
            return nullptr;
    }
}
