def set_params(params, id_optim):
    if id_optim is None:
        pass
    else:
        if id_optim == 0:
            params.dataset_nsamples = 5000
            params.epochs = 50
            params.seed = 42
            params.batch_size = 16
            params.test_split = 0.2
            params.val_split = 0.4
            params.backbone = 50
            params.dataset = "dataset/BigEarthNet_all_refactored_no_clouds_and_snow_v2_new_path.csv"
            params.path_nas = "/nas/softechict-nas-2/svincenzi/colorization_resnet/experiments_resnet50_AE/"
            params.log_dir = "Debug/"
            params.augmentation = 1
            params.decoder_version = 18
            params.pretrained = 0
            params.input_channels = 9
            params.out_channels = 2
            params.lr = 0.01
            params.optim = "SGD"
            params.img_size = 128
            params.weight_rec_loss = 100.
            params.grad_loss = 0
            params.weight_grad_loss = 0.1
            params.loss = "L1"
            params.scheduler = 1
            params.sched_step = 40
            params.sched_type = "step"
            params.path_model_dict = ""
            params.load_checkpoint = 0
            params.dropout = 0.3
            params.num_workers = 0

        params.log_dir = params.log_dir + "_batch_" + str(params.batch_size)

    return params
