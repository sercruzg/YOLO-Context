#include "concat_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

concat_layer make_concat_layer(int batch, int h, int w, int c, int lindex, int flip, int c_sec)
{
    concat_layer l = {0};
    l.type = CONCAT;

    l.inputs = h*w*c;
    l.outputs = h*w*(c + c_sec);
    l.batch=batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.c_sec = c_sec;
    l.out_h = h;
    l.out_w = w;
    l.out_c = c + c_sec;
    
    l.flip = flip;
    
    l.pointLayer = lindex;

    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward = forward_concat_layer;
    l.backward = backward_concat_layer;

#ifdef GPU
    l.forward_gpu = forward_concat_layer_gpu;
    l.backward_gpu = backward_concat_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, l.outputs*batch);
#endif
    fprintf(stderr, "concat           %4d x%4d x%4d   ->  %4d x%4d x%4d\n", w, h, c, l.out_w, l.out_h, l.out_c);
//     fprintf(stderr, "concat                           2 x %4d  ->  %4d\n", inputs, outputs);
    return l;
}

void forward_concat_layer(concat_layer l, network_state state)
{
    
}

void backward_concat_layer(concat_layer l, network_state state)
{
    
}

