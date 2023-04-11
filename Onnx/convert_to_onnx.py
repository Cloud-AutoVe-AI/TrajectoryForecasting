from utils.functions import *
from helper import load_solvers


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', type=int, default=1331)
    parser.add_argument('--model_num', type=int, default=39)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--dataset_type', type=str, default='ETRI')
    parser.add_argument('--model_name', type=str, default='autove')
    parser.add_argument('--best_k', type=int, default=10)
    parser.add_argument('--max_num_agents', type=int, default=30)

    args = parser.parse_args()
    test(args)

def test(args):

    # parent of working directory is base
    abspath = os.path.dirname(os.path.realpath(__file__))
    os.chdir(Path(abspath).parent.absolute())

    # CUDA setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_num))

    # type definition
    long_dtype, float_dtype = get_dtypes(useGPU=True)

    # path to saved network
    folder_name = args.dataset_type + '_' + args.model_name + '_model' + str(args.exp_id)
    path = os.path.join('./saved_models/', folder_name)

    # load parameter setting
    with open(os.path.join(path, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    print_training_info(saved_args)


    # define network
    solver = load_solvers(saved_args, 1000, float_dtype)
    solver.convert_to_onnx_file_CNN(args, abspath)
    solver.convert_to_onnx_file_TrajDec(args, abspath)


if __name__ == '__main__':
    main()


