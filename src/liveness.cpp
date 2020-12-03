//
// Created by ay27 on 9/14/17.
//

#include <liveness.h>

namespace SuperNeurons {

template<class value_type>
std::vector<std::pair<int, net_comp> > &
liveness_analysis_t<value_type>::get_subsequent_layers(int curt_layer_id, net_comp dir) {
    if (dir == FORWARD) {
        return subsequent_forward->operator[]((size_t) curt_layer_id);
    } else {
        return subsequent_backward->operator[]((size_t) curt_layer_id);
    }
}

template<class value_type>
bool liveness_analysis_t<value_type>::is_used_by_layer(int layer_id, net_comp dir, tensor_t<value_type> *t) {
    std::vector<tensor_t<value_type> *> *tensors = NULL;
    if (dir == FORWARD) {
        tensors = reg->get_forward_dependency(layer_id);
    } else if (dir == BACKWARD) {
        tensors = reg->get_backward_dependency(layer_id);
    }
    if (tensors == NULL) return false;
    for (size_t i = 0; i < tensors->size(); i++) {
        if (tensors->operator[](i) == t) {
            return true;
        }
    }
    return false;
}

template<class value_type>
bool liveness_analysis_t<value_type>::is_freeable_afterwards(int curt_layer_id, net_comp dir, tensor_t<value_type> *t) {
    // Get the subsequent layers in the direction given from the current layer. 
    std::vector<std::pair<int, net_comp> > subsequent_layers = get_subsequent_layers(curt_layer_id, dir);
    printf("Layer being checked is: %d %d\n", curt_layer_id, dir);
    for (size_t i = 0; i < subsequent_layers.size(); i++) {
        std::pair<int, net_comp> layer = subsequent_layers[i];
        printf("Subsequent layer: %d %d\n",layer.first, layer.second);
        // Check a tensor for each subsequent layer if it is being used and free it if it is.
        bool is_used = is_used_by_layer(layer.first, layer.second, t);
        if (is_used) {
            return false;
        }
    }
    return true;
}

template<class value_type>
bool liveness_analysis_t<value_type>::is_compressible_afterwards(int curt_layer_id, tensor_t<value_type> *t) {
    int i = 0; 
    net_comp dir = FORWARD;
    std::vector<std::pair<int, net_comp> > subsequent_layers = get_subsequent_layers(curt_layer_id, FORWARD);
    bool is_used = false; 
    while(i < subsequent_layers.size() && dir == FORWARD) {
        std::pair<int, net_comp> layer = subsequent_layers[i];
        // Next loop for backward
        if(layer.second == BACKWARD)
            break; 

        bool is_used = is_used_by_layer(layer.first, layer.second, t);
        // return false if we need to use it for a forward layer. 
        if(is_used)
            return false;
        
        ++i;
    } 
    is_used = false;
    printf("Checking backward!\n"); 
    while(i < subsequent_layers.size()) {
        std::pair<int, net_comp> layer = subsequent_layers[i];
        
        // here dependency will be backward. 
        bool is_used = is_used_by_layer(layer.first, layer.second, t);
        
        // We need it in backward so compress it.
        if(is_used)
            break;
        
        ++i;
    } 
    
    printf("Finished backward %d\n", is_used); 
    if(!is_used)
        return false; 

    return true;
}

// This is used to track which regulated tensors do each layers use.
template<class value_type>
void liveness_analysis_t<value_type>::set_ins(std::vector<std::vector<void *> > *ins, net_comp dir) {
    ins->resize(max_layer_id + 1);

    auto all_layers = reg->get_net_layers();
    auto nets = reg->get_net_comp_route();
    for (auto layer = nets.begin(); layer != nets.end(); ++layer) {
        if (dir != layer->second) {
            continue;
        }

        int layer_id = layer->first;
        ins->operator[](layer_id).resize(0);

        // we don't care about the DATA layer
        auto tmp = all_layers.find(layer_id);
        if (tmp == all_layers.end() || ((base_layer_t<value_type> *) tmp->second)->get_layer_type() == DATA_L) {
            continue;
        }

        std::vector<tensor_t<value_type> *> *tensors = NULL;
        if (dir == FORWARD) {
            tensors = reg->get_forward_dependency(layer_id);
        } else if (dir == BACKWARD) {
            tensors = reg->get_backward_dependency(layer_id);
        }
        if (tensors == NULL) return;

        for (size_t i = 0; i < tensors->size(); i++) {
            tensor_t<value_type> *t = tensors->operator[](i);
            if (t->get_type() != DATA && t->get_type() != CONV_BUFF) {
                continue;
            }
            auto r_it = regulated_tensors->find(t);
            if (r_it != regulated_tensors->end()) {
                ins->operator[](layer_id).push_back((void*)t);
            }
        }
    }
}

template<class value_type>
void liveness_analysis_t<value_type>::set_outs(std::vector<std::vector<void *> > *outs, net_comp dir) {
    outs->resize(max_layer_id + 1);

    // the free_list scan algorithm is quite simple and stupid, can we make it more elegant?

    auto all_layers = reg->get_net_layers();
    auto nets = reg->get_net_comp_route();

    for (auto layer = nets.begin(); layer != nets.end(); ++layer) {
        if (dir != layer->second) {
            continue;
        }

        int layer_id = layer->first;
        outs->operator[](layer_id).resize(0);

        // we don't care about DATA layer
        auto tmp = all_layers.find(layer_id);
        if (tmp == all_layers.end() || ((base_layer_t<value_type> *) tmp->second)->get_layer_type() == DATA_L) {
            continue;
        }

        for (auto it = regulated_tensors->begin(); it != regulated_tensors->end(); it++) {
            tensor_t<value_type> *t = (tensor_t<value_type> *) it->first;

            // ignore tensor in future
            if (dir == FORWARD && t->get_layer_id() > layer_id) {
                continue;
            }
            if (dir == BACKWARD && t->get_layer_id() < layer_id) {
                continue;
            }

#ifdef RECOMPUTE_ON
            if (dir == FORWARD) {
                base_layer_t<value_type>* l = (base_layer_t<value_type>*)(reg->get_net_layers().find(layer_id)->second);
                if (!l->get_next().empty()) {
                    if (is_checkpoint(l) && (t == reg->get_reg_output(layer_id, l->get_next()[0]->get_base_id()))) {
                        continue;
                    }
                }
            }
#endif
            // This method can potentially be used to decide what to compress.
            bool freeable = is_freeable_afterwards(layer_id, dir, t);

            if (freeable) {
                outs->operator[](layer_id).push_back((void *) t);
            }
        }
    }

    // purify outs, basically don't double free, checks all layers to see the same tensors don't happen twice.
    for (auto layer = nets.begin(); layer != nets.end(); ++layer) {
        if (dir != layer->second) {
            continue;
        }

        int layer_id = layer->first;

        for (auto it1 = outs->operator[](layer_id).begin(); it1 != outs->operator[](layer_id).end();) {

            bool should_delete = false;

            int start, end;
            if (dir == FORWARD) {
                start = 1; end = layer_id-1;
            } else {
                start = layer_id+1; end = max_layer_id;
            }
            for (int l = start; l <= end; ++l) {
                for (auto it2 = outs->operator[](l).begin(); it2 != outs->operator[](l).end(); ++it2) {
                    if (*it1 == *it2) {
                        should_delete = true;
                        break;
                    }
                }
                if (should_delete) {
                    break;
                }
            }

            if (should_delete) {
                it1 = outs->operator[](layer_id).erase(it1);
            } else {
                ++it1;
            }
        }
    }
}

/*
    Liveness based compression. We compress all the maps which are data and not going to be used in the forward pass.
*/
template<class value_type>
void liveness_analysis_t<value_type>::set_compress(std::vector<std::vector<void *> > *compress, net_comp dir) {
    if(dir == BACKWARD) 
        return;

    compress->resize(max_layer_id + 1);
    printf("Starting the compress now!\n"); 
    auto all_layers = reg->get_net_layers();
    auto nets = reg->get_net_comp_route();

     for (auto layer = nets.begin(); layer != nets.end(); ++layer) {
        // Make sure direction is forward.
        if (dir != layer->second) {
            continue;
        }
	printf("Accessing compress! \n");
        int layer_id = layer->first;
        compress->operator[](layer_id).resize(0);
	printf("Accessed compress! \n");
        // we don't care about the DATA layer
        auto tmp = all_layers.find(layer_id);
        if (tmp == all_layers.end() || ((base_layer_t<value_type> *) tmp->second)->get_layer_type() == DATA_L) {
            continue;
        }

        // Get forward tensors. 
        std::vector<tensor_t<value_type> *> *tensors = reg->get_forward_dependency(layer_id);

        if (tensors == NULL)
            return;

        for (size_t i = 0; i < tensors->size(); i++)
        {
            tensor_t<value_type> *t = tensors->operator[](i);
            // We just care to compress the data outs of each.
            if (t->get_type() != DATA)
            {
                continue;
            }
            
            auto r_it = regulated_tensors->find(t);
            if (r_it != regulated_tensors->end())
            {
                bool canCompress = is_compressible_afterwards(layer_id, t);
                if(canCompress) {
                    printf("Compress tensor %d at layer %d\n", t->get_layer_id(), layer_id);
                    compress->operator[](layer_id).push_back((void *)t);
                }
                    
                // ins->operator[](layer_id).push_back((void *)t);
            }
        }
    }

}


template<class value_type>
void liveness_analysis_t<value_type>::stash(int layer_id, net_comp dir) {
    //    std::vector<tensor_t<value_type>* >* tensors = NULL;
//    if (dir == FORWARD) {
//        tensors = reg->get_forward_dependency(layer_id);
//    } else if(dir == BACKWARD) {
//        tensors = reg->get_backward_dependency(layer_id);
//    }
//    if( tensors == NULL ) return;
//    /*------------------------------------------*/
//    //we get the tensors ready in the curt layers
//    for( size_t i = 0; i < tensors->size(); i++ ) {
//        tensor_t<value_type>* t = tensors->operator[](i);
//        typename std::map<tensor_t<value_type>*, mem_mode>::iterator it = regulated_tensors.find(t);
//        if (it != regulated_tensors.end()) {
//            if(t->get_state() == VOID) {
//                t->atomic_set_state( GPU );
//                t->stash_gpu_space();
//            }
//        }
//    }
//    return;
    std::vector<std::vector<tensor_t<value_type> *> > *ins=NULL;
    if (dir == FORWARD) {
        ins = (std::vector<std::vector<tensor_t<value_type> *> > *) &f_stash_tensors;
    } else if (dir == BACKWARD) {
        ins = (std::vector<std::vector<tensor_t<value_type> *> > *) &b_stash_tensors;
    }

    for (auto it = ins->operator[](layer_id).begin(); it != ins->operator[](layer_id).end(); ++it) {
        tensor_t<value_type> *t = *it;
#ifdef RECOMPUTE_ON
        // leave recompute tensor for recompute routine
        if (t->get_state() == RECOMPUTE) {
            continue;
        }
#endif
        if (t->get_type() == CONV_BUFF) {
	    // printf("Stashing space for conv_buff\n");
            // Here if compression is turned on we can instead decompress?
            // instead of stashing.
            t->stash_gpu_space();
        } else {
            // Instead of this we can do compress to GPU. This is purely for CPU to GPU. 
            // Let's separate the concerns. 
            t->CPUtoGPU();
        }
    }

    // Decompress compressed tensors on backward. 

    


#ifdef MEM_DEBUG
    printf("\n-------ins------\n");
    for (int i = 1; i < ins->size(); ++i) {
        printf("---layer %d\n", i);
        for (auto it = ins->operator[](i).begin(); it!=ins->operator[](i).end(); ++it) {
            printf("%p ", (*it));
        }
        printf("\n");
    }
#endif
}

template<class value_type>
void liveness_analysis_t<value_type>::update(int layer_id, net_comp dir) {
    //    // we only do free and offload to cpu here
//    typename std::map<tensor_t<value_type>*, mem_mode>::iterator it = regulated_tensors.begin();
//    for ( it = regulated_tensors.begin(); it != regulated_tensors.end(); it++ ) {
//        // we update the table, and set tensors that no longer to be used to VOID to save memory
//        // therefore, the operation is toward GPU tensors only
//        tensor_t<value_type>* t = it->first;
//        if( t->get_state() == VOID ) continue;
//
//        std::pair<bool, int> sta = is_freeable_afterwards(layer_id, dir, t);
//
//        if( t->get_state() == GPU ) {
//            // to check if the tensor is freeable
//            if(sta.first == true) {
//                t->atomic_set_state( VOID );
//#ifdef MEM_DEBUG
//                printf("tensor:%p is ready to free\n", t);
//#endif
//                t->free_gpu_space();
//            }
//        }
//    }
//    return;

    std::vector<std::vector<tensor_t<value_type> *> > *outs;
    if (dir == FORWARD) {
        outs = (std::vector<std::vector<tensor_t<value_type> *> > *) &f_free_tensors;
    } else {
        outs = (std::vector<std::vector<tensor_t<value_type> *> > *) &b_free_tensors;
    }

    for (auto it = outs->operator[](layer_id).begin(); it != outs->operator[](layer_id).end(); ++it) {
        tensor_t<value_type> *t = *it;
        t->free_gpu_space(VOID);
    }

    // Compress tensors for forward.
    
#ifdef MEM_DEBUG
    printf("\n-------outs------\n");
    for (int i = 1; i < outs->size(); ++i) {
        printf("---layer %d\n", i);
        for (auto it = outs->operator[](i).begin(); it!=outs->operator[](i).end(); ++it) {
            printf("%p ", (*it));
        }
        printf("\n");
    }
#endif
}


INSTANTIATE_CLASS(liveness_analysis_t);

} // namespace SuperNeurons
