#ifndef CONCAT_LAYER_H
#define CONCAT_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer concat_layer;

concat_layer make_concat_layer(int batch, int h, int w, int c, int l, int flip, int c_sec);

void forward_concat_layer(concat_layer layer, network_state state);
void backward_concat_layer(concat_layer layer, network_state state);

#ifdef GPU
void forward_concat_layer_gpu(concat_layer layer, network_state state) ;
void backward_concat_layer_gpu(concat_layer layer, network_state state);
#endif

#endif

