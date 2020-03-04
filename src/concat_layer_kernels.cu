#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef CUDNN
#pragma comment(lib, "cudnn.lib")  
#endif

extern "C" {
#include "concat_layer.h"
#include "blas.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void apply_concat_gpu(float* data_im, float* data_imContext, float* data_imJoint,
        const int height, const int width, const int channel, const int size, const int flip, const int channel_sec) {

    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (index >= size)
        return;

    int col = (index % width);
    index = (index / width);
    int row = (index % height);
    index = (index / height);

    int col_out;
    if(flip)
        col_out = width - col - 1;
    else
        col_out = col;

    for (int c = 0; c < channel; c++){
        data_imJoint[col + width*(row + height*c)]  = data_im[col + width*(row + height*c)];
        //data_imJoint[col_out + width*(row + height*(c+channel))]  = data_imContext[col + width*(row + height*c)];
    }
    for (int c = 0; c < channel_sec; c++){
        //data_imJoint[col + width*(row + height*c)]  = data_im[col + width*(row + height*c)];
        data_imJoint[col_out + width*(row + height*(c+channel))]  = data_imContext[col + width*(row + height*c)];
    }
}

void forward_concat_layer_gpu(concat_layer l, network_state state) {

    float *objInput = state.objNet.layers[state.objNet.n - 1 + l.pointLayer].output_gpu;
    float *contInput = state.contNet.layers[state.contNet.n - 1 + l.pointLayer].output_gpu;
    //fprintf(stderr, "N= %d,  point=%d select=%d\n",state.objNet.n, l.pointLayer,state.objNet.n - 1 + l.pointLayer);
    int total = l.h * l.w;
    int flip = l.flip;
    for(int i = 0; i < l.batch; i++){
        float *a = objInput + i*l.c*l.h*l.w;
        float *b = contInput + i*l.c_sec*l.h*l.w;
        float *c = l.output_gpu + i*l.out_h*l.out_w*l.out_c;
        apply_concat_gpu<<<cuda_gridsize(total), BLOCK>>>(a, b, c, l.h, l.w, l.c, total, flip, l.c_sec);
    }

}
void backward_concat_layer_gpu(concat_layer l, network_state state)
{
}