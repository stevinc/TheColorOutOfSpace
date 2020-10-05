def set_params(params, id_optim):
    if id_optim is None:
        pass
    else:
        if id_optim == 0:
            params.dataset_nsamples = 5000
            params.seed = 19
            params.epochs = 30
            params.batch_size = 32
            params.resnet_version = 18
            params.pretrained = 0
            params.out_cls = 19
            params.dataset = "dataset/BigEarthNet_all_refactored_no_clouds_and_snow_v2_new_path_new_labels.csv"
            params.log_dir = "/9_bands/19Labels/5k_R18/exp_1"
            params.bands = [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
            params.input_channels = 9
            params.lr = 0.1
            params.optim = "SGD"
            params.img_size = 128
            params.scheduler = 1
            params.sched_type = "multi"
            params.sched_milestones = [10, 40]
            params.load_checkpoint = 0
            params.path_model_dict = "/nas/softechict-nas-2/svincenzi/colorization_resnet/experiments_resnet18_AE/500k_augmentation_scratch_continue_training/_batch_16/last.pth.tar"
            params.load_checkpoint_tr = 0
            params.path_model_dict_tr = ""
            params.num_workers = 4
        elif id_optim == 1:
            params.dataset_nsamples = 5000
            params.seed = 19
            params.epochs = 30
            params.batch_size = 32
            params.resnet_version = 18
            params.pretrained = 0
            params.out_cls = 19
            params.dataset = "dataset/BigEarthNet_all_refactored_no_clouds_and_snow_v2_new_path_new_labels.csv"
            params.log_dir = "/3_bands/19Labels/5k_R18/exp_1"
            params.bands = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            params.input_channels = 3
            params.lr = 0.1
            params.optim = "SGD"
            params.img_size = 128
            params.scheduler = 1
            params.sched_type = "multi"
            params.sched_milestones = [10, 40]
            params.load_checkpoint = 0
            params.path_model_dict = "/nas/softechict-nas-2/svincenzi/colorization_resnet/experiments_resnet18_AE/500k_augmentation_scratch_continue_training/_batch_16/last.pth.tar"
            params.load_checkpoint_tr = 0
            params.path_model_dict_tr = ""
            params.num_workers = 4

        params.log_dir = params.log_dir + "_batch_" + str(params.batch_size)

    return params


