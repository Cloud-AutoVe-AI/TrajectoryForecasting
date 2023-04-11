from utils.libraries import *
from torch.utils.data import DataLoader
from utils.collate import *
from utils.functions import read_config

LOADER_TYPE_A = ['ilvm', 'esp', 'desire']
LOADER_TYPE_B = ['covernet']
LOADER_TYPE_Cps = ['baseline', 'baseline_nf','scratch_t3']
LOADER_TYPE_D = ['autove']
LOADER_TYPE_E = ['autove_ped']
LOADER_TYPE_F = ['NMP']
LOADER_TYPE_G = ['agentformer'] # update, 220415

def load_datasetloader(args, dtype, isTrain=True):

    config = read_config()

    # update, 220415
    # model_mode = args.model_mode
    # if (model_mode == 'pedestrian' and args.dataset_type in ['argoverse']):
    #     sys.exit("[Error] '%s' mode 'is not supported with the dataset {%s} !!" % (args.model_mode, args.dataset_type))
    #
    # if (model_mode == 'pedestrian' and args.model_name not in ['autove_ped']):
    #     sys.exit("[Error] '%s' mode is not supported by the model {%s} !!" % (args.model_mode, args.model_name))

    # --------------------------
    # Argoverse
    # --------------------------
    if (args.dataset_type == 'argoverse'):
        if (args.model_name in LOADER_TYPE_Cps):
            from ArgoverseDataset.loader_typeCps import DatasetLoader
            seq_collate = seq_collate_typeCps
        else:
            sys.exit("[Error] No loader type exists for '%s' in 'Argoverse' !!" % args.model_name)
        args.dataset_path = config['Argoverse_Forecasting']['dataset_path']


    # --------------------------
    # Nuscenes
    # --------------------------
    #
    #            |  A  |  B  |  Cps |  D  |  E  |
    # -------------------------------------------
    # vehicle    |  O  |  O  |   O  |  O  |  X  |
    # pedestrian |  X  |  X  |   X  |  X  |  O  |
    #
    # --------------------------
    elif (args.dataset_type == 'nuscenes'):
        if (args.model_name in LOADER_TYPE_A):
            from NuscenesDataset.loader_typeA import DatasetLoader
            seq_collate = seq_collate_typeA
        elif (args.model_name in LOADER_TYPE_B):
            from NuscenesDataset.loader_typeB import DatasetLoader
            seq_collate = seq_collate_typeB
        elif (args.model_name in LOADER_TYPE_Cps):
            from NuscenesDataset.loader_typeCps import DatasetLoader
            seq_collate = seq_collate_typeCps
        elif (args.model_name in LOADER_TYPE_D):
            from NuscenesDataset.loader_typeD import DatasetLoader
            seq_collate = seq_collate_typeD
        elif (args.model_name in LOADER_TYPE_E):
            from NuscenesDataset.loader_typeE import DatasetLoader
            seq_collate = seq_collate_typeD
        elif (args.model_name in LOADER_TYPE_F):
            from NuscenesDataset.loader_typeF import DatasetLoader
            seq_collate = seq_collate_typeF
        else:
            sys.exit("[Error] No loader type exists for '%s' in 'Nuscenes' !!" % args.model_name)
        args.dataset_path = config['Nuscenes']['dataset_path']

    # --------------------------
    # Voss
    # --------------------------
    elif (args.dataset_type == 'ETRI'):
        if (args.model_name in LOADER_TYPE_D):
            from ETRIDataset.loader_typeD import DatasetLoader
            seq_collate = seq_collate_typeD
        elif (args.model_name in LOADER_TYPE_E):
            from ETRIDataset.loader_typeE import DatasetLoader
            seq_collate = seq_collate_typeE
        else:
            sys.exit("[Error] No loader type exists for '%s' in 'ETRIDataset' !!" % args.model_name)
        args.dataset_path = config['ETRI']['dataset_path']

    # ---------------------------
    # ETH/UCY
    # ---------------------------
    # update, 220415
    elif (args.dataset_type == 'ETHUCY'):
        if (args.model_name in LOADER_TYPE_G):
            from ETHUCY.loader_typeG import DatasetLoader
            seq_collate = seq_collate_typeG
        else:
            sys.exit("[Error] No loader type exists for '%s' in 'ETH/UCY' !!" % args.model_name)
        args.dataset_path = config['ETHUCY']['dataset_path']

    else:
        sys.exit("[Error] '%s' dataset is not supported !!" % args.dataset_path)

    # prepare data
    dataset_loader = DatasetLoader(args=args, isTrain=isTrain, dtype=dtype)

    if (isTrain==False):
        return dataset_loader, 0
    else:
        data_loader = DataLoader(dataset_loader, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_cores, drop_last=True, collate_fn=seq_collate)
        return dataset_loader, data_loader

def load_solvers(args, num_train_scenes, dtype):

    if (args.model_name == 'ilvm'):
        from optimization.ilvm_solver import Solver
        return Solver(args, num_train_scenes, dtype)
    elif (args.model_name == 'covernet'):
        from optimization.covernet_solver import Solver
        return Solver(args, num_train_scenes, dtype)
    elif (args.model_name == 'autove'):
        from optimization.autove_solver import Solver
        return Solver(args, num_train_scenes, dtype)
    elif (args.model_name == 'autove_ped'):
        from optimization.autove_ped_solver import Solver
        return Solver(args, num_train_scenes, dtype)
    elif (args.model_name == 'scratch_t3'):
        from optimization.scratch_t3_solver import Solver
        return Solver(args, num_train_scenes, dtype)
    elif (args.model_name == 'baseline'):
        from optimization.baseline_solver import Solver
        return Solver(args, num_train_scenes, dtype)
    elif (args.model_name == 'baseline_nf'):
        from optimization.baseline_nf_solver import Solver
        return Solver(args, num_train_scenes, dtype)
    elif (args.model_name == 'NMP'):
        from optimization.NMP_solver import Solver
        return Solver(args, num_train_scenes, dtype)
    elif (args.model_name == 'agentformer'): # update, 220415
        from optimization.AF_solver import Solver
        return Solver(args, num_train_scenes, dtype)
    else:
        sys.exit("[Error] There is no solver for '%s' !!" % args.model_name)
