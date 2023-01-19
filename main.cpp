#include<stdio.h>
#include<math.h>
#include<algorithm>
#include<float.h>
#include<chrono>
using namespace std;
using namespace chrono;

// copy from ncnn: https://github.com/Tencent/ncnn/blob/master/src/layer/softmax.cpp#L58
void softmax_e(float* ptr, int size)
{
    float max = -FLT_MAX;
    for (int i = 0; i < size; i++)
    {
        max = std::max(max, ptr[i]);
    }

    float sum = 0.f;
    for (int i = 0; i < size; i++)
    {
        ptr[i] = expf(ptr[i] - max);
        sum += ptr[i];
    }

    for (int i = 0; i < size; i++)
    {
        ptr[i] /= sum;
    }
}

// for compare
void softmax_2(float* ptr, int size)
{
    float max = -FLT_MAX;
    for (int i = 0; i < size; i++)
    {
        max = std::max(max, ptr[i]);
    }

    float sum = 0.f;
    for (int i = 0; i < size; i++)
    {
        ptr[i] = exp2f(ptr[i] - max);
        sum += ptr[i];
    }

    for (int i = 0; i < size; i++)
    {
        ptr[i] /= sum;
    }
}

// idea from paper: https://www.nature.com/articles/s41598-021-94691-7
void fast_base2(unsigned int* inptr, float* outptr, int size)
{
    const unsigned int mant_mask = 0b11111111111111111111111;
    const unsigned int exp_mask = 0b00111111100000000000000000000000;

    unsigned int bits;
    float* tmp_float = (float*)&bits;

    float sum = 0.f;
    for (int i = 0; i < size; i++)
    {
        bits = (inptr[i] | (1 << 7)) << 23;
        sum += *tmp_float;
    }

    // get the exp and mant of sum
    unsigned int exp_sum = *(unsigned int*)&sum >> 23 & 0b01111111;
    unsigned int mant_sum_uint = (*(unsigned int*)&sum & mant_mask) | exp_mask;
    float* p_mant_sum_fp = (float*)&mant_sum_uint;

    // inverse the mant
    *p_mant_sum_fp = 1.f / *p_mant_sum_fp;

    // get the exp and mant of inverse_mant
    int exp_imant = *(unsigned int*)p_mant_sum_fp >> 23 & 0b01111111;
    unsigned int mant_imant_uint = (*(unsigned int*)p_mant_sum_fp & mant_mask);

    exp_imant = exp_imant - exp_sum;

    for (int i = 0; i < size; i++)
    {
        bits = (exp_imant + inptr[i]) << 23 | mant_imant_uint;
        outptr[i] = *tmp_float;
    }
}

int main(void)
{
    const int size = 256*256*256;

    int *src = new int[size];
    float *naive = new float[size];
    float *base2 = new float[size];
    unsigned int *infast = new unsigned int[size];
    float *outfast = new float[size];

    // copy data
    for(int i = 0; i < size; i++)
    {
        int rnd = i % 125;
        src[i] = rnd;
        naive[i] = rnd;
        base2[i] = rnd;
        infast[i] = rnd;
    }
    
    auto time0 = high_resolution_clock::now();

    // naive softmax
    softmax_e(naive, size);
    auto time1 = high_resolution_clock::now();
    // base2 softmax
    softmax_2(base2, size);
    auto time2 = high_resolution_clock::now();
    // fast base2 softmax
    fast_base2(infast, outfast, size);
    auto time3 = high_resolution_clock::now();

    // compare
    float e1 = 0.f, e2 = 0.f;
    for(int i = 0; i < size; i++)
    {
        // printf("%02d: %03d %.2f, %.2f, %.2f\n", i, src[i], naive[i], base2[i], outfast[i]);
        e1 += abs(naive[i] - outfast[i]);
        e2 += abs(base2[i] - outfast[i]);
    }
    e1 /= size;
    e2 /= size;
    printf("e1:%.2f, e2:%.2f\n", e1, e2);

    // speed
    auto t1 = duration_cast<microseconds>(time1 - time0);
    auto t2 = duration_cast<microseconds>(time2 - time1);
    auto t3 = duration_cast<microseconds>(time3 - time2);
    printf("t1:%lld, t2:%lld, t3:%lld\n", t1.count(), t2.count(), t3.count());

    // free memory
    delete[] naive;
    delete[] base2;
    delete[] infast;
    delete[] outfast;

    return 0;
}