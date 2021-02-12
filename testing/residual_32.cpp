#include <stdlib.h>
#include<math.h>
#include <superneurons.h>
#include <solver.h>
#include <sched.h>
using namespace SuperNeurons;


base_layer_t<float> *residual_block(base_layer_t<float>* bottom, size_t out_num, bool increase_dim, bool first=false) {

    size_t stride = 1;
    if (increase_dim) {
        stride = 2;
    }

    base_layer_t<float>* fork = (base_layer_t<float>*) new fork_layer_t<float>();
    bottom->hook(fork);

    base_layer_t<float> *left = fork, *right = fork;

    if ( !first ) {
        base_layer_t<float>* bn = (base_layer_t<float> *) new batch_normalization_layer_t<float>(CUDNN_BATCHNORM_SPATIAL, 0.001);
        base_layer_t<float>* act = (base_layer_t<float> *) new act_layer_t<float>();
        fork->hook(bn);
        bn->hook(act);

        left = act;
    }

    base_layer_t<float>* conv1 = (base_layer_t<float>*) new conv_layer_t<float>(out_num, 3, stride, 1, 1, new xavier_initializer_t<float>(), true);
    base_layer_t<float>* bn1 = (base_layer_t<float>*) new batch_normalization_layer_t<float>(CUDNN_BATCHNORM_SPATIAL, 0.001);
    base_layer_t<float>* act1 = (base_layer_t<float> *) new act_layer_t<float>();
    base_layer_t<float>* conv2 = (base_layer_t<float>*) new conv_layer_t<float>(out_num, 3, 1, 1, 1, new xavier_initializer_t<float>(), true);

    left->hook(conv1);
    conv1->hook(bn1);
    bn1->hook(act1);
    act1->hook(conv2);

    base_layer_t<float>* join = (base_layer_t<float>*) new join_layer_t<float>();

    conv2->hook(join);

    if (increase_dim) {
        base_layer_t<float> *avg_pool = (base_layer_t<float> *) new pool_layer_t<float>(
                CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 2, 2);
        base_layer_t<float> *padding = (base_layer_t<float> *) new padding_layer_t<float>(out_num / 4, 0, 0);

        right->hook(avg_pool);
        avg_pool->hook(padding);
        padding->hook(join);
    } else {
        right->hook(join);
    }
    return join;
}

int main(int argc, char **argv) {
    cpu_set_t mask; 
    CPU_ZERO(&mask);
    CPU_SET(1, &mask);
    int result = sched_setaffinity(0, sizeof(mask), &mask);
    char *train_label_bin;
    char *train_image_bin;
    char *test_label_bin;
    char *test_image_bin;
    char *train_mean_file;

    int loop_num = 5;

    base_solver_t<float>* solver = (base_solver_t<float>*) new nesterov_solver_t<float>(0.01, 0.0004, 0.9);
    solver->set_lr_decay_policy(ITER, {400, 50000, 100000}, {0.1, 0.01, 0.001});

    network_t<float> n(solver);

train_mean_file = (char *) "/home/shauryakamle/superneurons-release/cifar-10-batches-bin/cifar_train.mean";
    train_image_bin = (char *) "/home/shauryakamle/superneurons-release/cifar-10-batches-bin/cifar10_train_image_0.bin";
    train_label_bin = (char *) "/home/shauryakamle/superneurons-release/cifar-10-batches-bin/cifar10_train_label_0.bin";
    test_image_bin  = (char *) "/home/shauryakamle/superneurons-release/cifar-10-batches-bin/cifar10_test_image_0.bin";
    test_label_bin  = (char *) "/home/shauryakamle/superneurons-release/cifar-10-batches-bin/cifar10_test_label_0.bin";
    /*train_mean_file = (char *) "/data/lwang53/cifar/cifar_train.mean";
    train_image_bin = (char *) "/data/lwang53/cifar/cifar10_train_image_0.bin";
    train_label_bin = (char *) "/data/lwang53/cifar/cifar10_train_label_0.bin";
    test_image_bin  = (char *) "/data/lwang53/cifar/cifar10_test_image_0.bin";
    test_label_bin  = (char *) "/data/lwang53/cifar/cifar10_test_label_0.bin"; */

    const size_t batch_size = 100; //train and test must be same
    const size_t C = 3, H = 32, W = 32;


    base_preprocess_t<float> *mean_sub =
            (base_preprocess_t<float> *) new mean_subtraction_t<float>(batch_size, C, H, W, train_mean_file);

    base_preprocess_t<float> *pad = (base_preprocess_t<float> *) new border_padding_t<float>(
            batch_size, C, H, W, 4, 4);
    base_preprocess_t<float> *crop = (base_preprocess_t<float> *) new random_crop_t<float>(
            batch_size, C, H + 8, W + 8, batch_size, C, H, W);
    base_preprocess_t<float> *flip = (base_preprocess_t<float> *) new random_flip_left_right_t<float>(
            batch_size, C, H, W);
    base_preprocess_t<float> *bright = (base_preprocess_t<float> *) new random_brightness_t<float>(
            batch_size, C, H, W, 63);
    base_preprocess_t<float> *contrast = (base_preprocess_t<float> *) new random_contrast_t<float>(
            batch_size, C, H, W, 0.2, 1.8);
    base_preprocess_t<float> *standardization =
            (base_preprocess_t<float> *) new per_image_standardization_t<float>(
                    batch_size, C, H, W);

    base_preprocess_t<float> *mean_sub1 =
            (base_preprocess_t<float> *) new mean_subtraction_t<float>(batch_size, C, H, W, train_mean_file);
    base_preprocess_t<float> *standardization1 =
            (base_preprocess_t<float> *) new per_image_standardization_t<float>(
                    batch_size, C, H, W);


    preprocessor<float> *processor = new preprocessor<float>();
    processor->add_preprocess(mean_sub)
            ->add_preprocess(pad)
            ->add_preprocess(crop)
            ->add_preprocess(flip)
            ->add_preprocess(bright)
            ->add_preprocess(contrast)
            ->add_preprocess(standardization);

    preprocessor<float>* p2 = new preprocessor<float>();
    p2->add_preprocess(new mean_subtraction_t<float>(batch_size, C, H, W, train_mean_file))
            ->add_preprocess(new per_image_standardization_t<float>(batch_size, C, H, W));


    parallel_reader_t<float> reader2(test_image_bin, test_label_bin, 1, batch_size, C, H, W, p2, 10, 1);
    base_layer_t<float> *data_2 = (base_layer_t<float> *) new data_layer_t<float>(DATA_TEST, &reader2);
    parallel_reader_t<float> reader1(train_image_bin, train_label_bin, 4, batch_size, C, H, W, processor, 10, 4);
    base_layer_t<float> *data_1 = (base_layer_t<float> *) new data_layer_t<float>(DATA_TRAIN, &reader1);

    //if the dims of H,W after conv is not reduced, pad with half the filter sizes (round down). 3/2 = 1.5 = 1;
    base_layer_t<float> *conv_1 = (base_layer_t<float> *) new conv_layer_t<float>(16, 3, 1, 1, 1, new xavier_initializer_t<float>(),
                                                                                  true);
    base_layer_t<float> *bn_1 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(CUDNN_BATCHNORM_SPATIAL,
                                                                                               0.001);
    base_layer_t<float> *act_1 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                CUDNN_NOT_PROPAGATE_NAN);


    base_layer_t<float> *bn_2 = (base_layer_t<float> *) new batch_normalization_layer_t<float>(CUDNN_BATCHNORM_SPATIAL, 0.001);
    base_layer_t<float> *act_2 = (base_layer_t<float> *) new act_layer_t<float>(CUDNN_ACTIVATION_RELU,
                                                                                CUDNN_NOT_PROPAGATE_NAN);

    base_layer_t<float> *pool_1 = (base_layer_t<float> *) new pool_layer_t<float>(
            CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 1, 1, 8, 8);

    base_layer_t<float> *full_conn_1 = (base_layer_t<float> *) new fully_connected_layer_t<float>(10, new xavier_initializer_t<float>(),
                                                                                                  true);
    base_layer_t<float> *softmax = (base_layer_t<float> *) new softmax_layer_t<float>(CUDNN_SOFTMAX_ACCURATE,
                                                                                      CUDNN_SOFTMAX_MODE_INSTANCE);

    //setup test
    data_2->hook_to(conv_1);
    //setup network
    data_1->hook(conv_1);
    conv_1->hook(bn_1);
    bn_1->hook(act_1);

    // change the loop_num to change the depth
    int nstage[6] = {16, 32, 64, 128, 256, 512};
//    int nstage[6] = {2, 4, 8, 16, 32, 64};

    base_layer_t<float> *net = act_1;

    net = residual_block(net, nstage[0], false, true);

    for (int i = 1; i < loop_num; i++) {
        net = residual_block(net, nstage[0], false);
    }


    net = residual_block(net, nstage[1], true);

    for (int i = 1; i < loop_num; i++) {
        net = residual_block(net, nstage[1], false);
    }


    net = residual_block(net, nstage[2], true);

    for (int i = 1; i < loop_num; i++) {
        net = residual_block(net, nstage[2], false);
    }

    net->hook(bn_2);
    bn_2->hook(act_2);
    act_2->hook(pool_1);
    pool_1->hook(full_conn_1);
    full_conn_1->hook(softmax);

    n.fsetup(data_1);
    n.bsetup(softmax);
    n.setup_test(data_2, 100);
    const size_t train_imgs = 50000;
    const size_t tracking_window = train_imgs / batch_size;

    n.train(200000, tracking_window, 500);

}

