from utils.functions import *
from utils.loss import calc_ED_error
from helper import load_datasetloader, load_solvers

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', type=int, default=1013)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='ETRI')
    parser.add_argument('--model_name', type=str, default='autove')
    parser.add_argument('--start_frm_idx', type=int, default=24) # update, 211223
    parser.add_argument('--step_size', type=int, default=10000)
    parser.add_argument('--best_k', type=int, default=10)
    parser.add_argument('--max_num_agents', type=int, default=30)
    parser.add_argument('--model_num', type=int, default=24)

    args = parser.parse_args()
    test(args)

def test(args):


    # parent of working directory is base
    abspath = os.path.dirname(os.path.realpath(__file__))
    os.chdir(Path(abspath).parent.absolute())

    # CUDA setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_num))

    # type definition
    _, float_dtype = get_dtypes(useGPU=True)

    # path to saved network
    folder_name = args.dataset_type + '_' + args.model_name + '_model' + str(args.exp_id)
    path = os.path.join('./saved_models/', folder_name)

    # load parameter setting
    with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    saved_args.best_k = args.best_k
    print_training_info(saved_args)

    # load test data
    data_loader, _ = load_datasetloader(args=saved_args, isTrain=False, dtype=torch.FloatTensor)

    # define network
    solver = load_solvers(saved_args, data_loader.num_test_scenes, float_dtype)
    solver.onnx_load_splited_modules(args.model_num)
    solver.read_onnx_files(args, './Onnx')


    save_dir = os.path.join(abspath, 'sample_bin')
    if (os.path.exists(save_dir)==False):
        os.mkdir(save_dir)

    for b in range(args.start_frm_idx, len(data_loader.test_data), args.step_size):
    # for b in range(0, 1):

        # data loading
        data = data_loader.next_sample(b, mode='test')

        # inference
        solver.onnx_make_bin_samples(data, args.max_num_agents, save_dir, b)

        d = 0;


if __name__ == '__main__':
    main()


