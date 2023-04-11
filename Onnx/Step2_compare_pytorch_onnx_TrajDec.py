from utils.functions import *
from utils.loss import calc_ED_error
from helper import load_datasetloader, load_solvers

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', type=int, default=300)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='ETRI')
    parser.add_argument('--model_name', type=str, default='autove_ped')
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--best_k', type=int, default=10)
    parser.add_argument('--max_num_agents', type=int, default=30)
    parser.add_argument('--model_num', type=int, default=6)

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
    saved_args.dataset_type = 'ETRI'
    print_training_info(saved_args)

    # load test data
    data_loader, _ = load_datasetloader(args=saved_args, isTrain=False, dtype=torch.FloatTensor)

    # define network
    solver = load_solvers(saved_args, data_loader.num_test_scenes, float_dtype)
    solver.onnx_load_splited_modules(args.model_num)
    solver.read_onnx_files(args, './Onnx')

    for b in range(0, len(data_loader.test_data), args.step_size):

        # data loading
        data = data_loader.next_sample(b, mode='test')

        # inference
        rt = solver.onnx_compare_pytorch_onnx_models(data, args.max_num_agents)



if __name__ == '__main__':
    main()


