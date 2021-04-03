#include <stdlib.h>
#include <superneurons.h>
using namespace SuperNeurons;

int main(int argc, char **argv) {
    
    char* train_label_bin;
    char* train_image_bin;
    char* train_mean_file;
    
    char* test_label_bin;
    char* test_image_bin;
    char* checkpoint_file;
    
    if( argc != 4 ) {
        printf("please run as ./alexnet batch_size training_iters do_compress\n");
        return 0;
    }
    
    train_mean_file = (char *) "/mnt/nfs/scratch1/shauryakamle/image-net/train_bin/train_mean.bin";
    train_image_bin = (char *) "/mnt/nfs/scratch1/shauryakamle/image-net/train_bin/data_train_data_0.bin";
    train_label_bin = (char *) "/mnt/nfs/scratch1/shauryakamle/image-net/train_bin/data_train_label_0.bin";
    test_image_bin  = (char *) "/mnt/nfs/scratch1/shauryakamle/image-net/val_bin/data_val_data_0.bin";
    test_label_bin  = (char *) "/mnt/nfs/scratch1/shauryakamle/image-net/val_bin/data_val_label_0.bin";
    checkpoint_file = (char *) "/mnt/nfs/scratch1/shauryakamle/image-net/train_bin/alexnet_checkpoint";
    
    const size_t batch_size = atoi(argv[1]); //train and test must be same
    const size_t H = 227, W = 227;
    
    preprocessor<float>* processor_train = new preprocessor<float>();
    processor_train->add_preprocess(new mean_subtraction_t<float>(batch_size, 3, 256, 256, train_mean_file));
    processor_train->add_preprocess(new random_crop_t<float>(batch_size, 3, 256, 256, batch_size, 3, H, W));
    processor_train->add_preprocess(new random_flip_left_right_t<float>(batch_size, 3, 227, 227));
    
    preprocessor<float>* processor_test = new preprocessor<float>();
    processor_test->add_preprocess(new mean_subtraction_t<float>(batch_size, 3, 256, 256, train_mean_file));
    processor_test->add_preprocess(new central_crop_t<float>(batch_size, 3, 256, 256, batch_size, 3, H, W));
    
    //test
    
    parallel_reader_t<float > reader2 (test_image_bin, test_label_bin, 1, batch_size, 3, H, W, processor_test, 1, 1);
    base_layer_t<float>*   data_2 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TEST, &reader2);
    //train
    parallel_reader_t<float > reader1 (train_image_bin, train_label_bin, 4, batch_size, 3, H, W, processor_train, 4, 2);
    base_layer_t<float>*   data_1 = (base_layer_t<float>*) new data_layer_t<float>(DATA_TRAIN, &reader1);
    
    /*--------------network configuration--------------*/
    base_solver_t<float>* solver = (base_solver_t<float> *) new momentum_solver_t<float>(0.01, 0.0005, 0.9);
    solver->set_lr_decay_policy(ITER, {100000, 200000, 300000, 400000}, {0.001, 0.0001, 0.00001, 0.000001});
    bool do_compress = atoi(argv[3]);
    network_t<float> n(solver, do_compress);
    
    //    network_saver* saver = new network_saver_impl<float>(checkpoint_file, n.get_registry());
    //    install_signal_processor(saver);
    
    base_layer_t<float>* conv_1 = (base_layer_t<float>*) new conv_layer_t<float>(96, 11, 4, 0, 0, new gaussian_initializer_t<float>(0, 0.01), true, new constant_initializer_t<float>(0.0));
    base_layer_t<float>* relu_1 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float>* lrn_1  = (base_layer_t<float>*) new LRN_layer_t<float>();
    base_layer_t<float>* pool_1 = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);
    
    base_layer_t<float>* conv_2 = (base_layer_t<float>*) new conv_layer_t<float>(256, 5, 1, 2, 2, new gaussian_initializer_t<float>(0, 0.01), true, new constant_initializer_t<float>(0.1));
    base_layer_t<float>* relu_2 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float>* lrn_2  = (base_layer_t<float>*) new LRN_layer_t<float>();
    base_layer_t<float>* pool_2 = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);
    
    base_layer_t<float>* conv_3 = (base_layer_t<float>*) new conv_layer_t<float>(384, 3, 1, 1, 1,new gaussian_initializer_t<float>(0, 0.01), true, new constant_initializer_t<float>(0.0));
    base_layer_t<float>* relu_3 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    
    base_layer_t<float>* conv_4 = (base_layer_t<float>*) new conv_layer_t<float>(384, 3, 1, 1, 1, new gaussian_initializer_t<float>(0, 0.01), true, new constant_initializer_t<float>(0.1));
    base_layer_t<float>* relu_4 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    
    base_layer_t<float>* conv_5 = (base_layer_t<float>*) new conv_layer_t<float>(256, 3, 1, 1, 1, new gaussian_initializer_t<float>(0, 0.01), true, new constant_initializer_t<float>(0.1));
    base_layer_t<float>* relu_5 = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float>* pool_5 = (base_layer_t<float>*) new pool_layer_t<float>(CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 3, 3);
    
    base_layer_t<float>* full_conn_1 = (base_layer_t<float>*) new fully_connected_layer_t<float> (4096,new gaussian_initializer_t<float>(0, 0.005), true,
                                                                                                  new constant_initializer_t<float>(0.1));
    base_layer_t<float>* relu_6      = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float>* drop_1      = (base_layer_t<float>*) new dropout_layer_t<float>(0.5);
    
    base_layer_t<float>* full_conn_2 = (base_layer_t<float>*) new fully_connected_layer_t<float> (4096, new gaussian_initializer_t<float>(0, 0.005), true,
                                                                                                  new constant_initializer_t<float>(0.1));
    base_layer_t<float>* relu_7      = (base_layer_t<float>*) new act_layer_t<float>(CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN);
    base_layer_t<float>* drop_2      = (base_layer_t<float>*) new dropout_layer_t<float>(0.5);
    
    base_layer_t<float>* full_conn_3 = (base_layer_t<float>*) new fully_connected_layer_t<float> (1000, new gaussian_initializer_t<float>(0, 0.01), true,
                                                                                                  new constant_initializer_t<float>(0.0));
    
    base_layer_t<float>* softmax     = (base_layer_t<float>*) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE);
    
    //setup test
    data_2->hook_to( conv_1 );
    //setup network
    data_1->hook( conv_1 );
    conv_1->hook( relu_1 );
    relu_1->hook( lrn_1  );
    lrn_1->hook(  pool_1 );
    
    pool_1->hook( conv_2 );
    conv_2->hook( relu_2 );
    relu_2->hook( lrn_2 );
    lrn_2->hook(  pool_2 );
    
    pool_2->hook( conv_3 );
    conv_3->hook( relu_3 );
    
    relu_3->hook( conv_4 );
    conv_4->hook( relu_4 );
    
    relu_4->hook( conv_5 );
    conv_5->hook( relu_5 );
    relu_5->hook( pool_5 );
    
    pool_5->hook( full_conn_1 );
    full_conn_1->hook( relu_6 );
    relu_6->hook( drop_1 );
    
    drop_1->hook( full_conn_2 );
    full_conn_2->hook( relu_7 );
    relu_7->hook( drop_2 );
    
    drop_2->hook( full_conn_3 );
    full_conn_3->hook( softmax );
    
    n.fsetup( data_1  );
    n.bsetup( softmax );
    //    saver->load();
    
    n.setup_test( data_2, 50000 / batch_size );
    
    const size_t train_imgs = 1281166;
    const size_t tracking_window = train_imgs / batch_size;
    
    // 200 epoch
    // testing every epoch
    fprintf(stderr, "total iteration: %zu, test interval : %zu\n", (train_imgs/batch_size)*200, train_imgs/batch_size);
    n.train(atoi(argv[2]), tracking_window, 1000);
}
