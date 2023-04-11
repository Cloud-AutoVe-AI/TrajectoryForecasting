from utils.libraries import *
import argumentparser as ap

def main():

    # parent of working directory is base
    abspath = os.path.dirname(os.path.realpath(__file__))
    os.chdir(Path(abspath).parent.absolute())

    # control argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default='nuscenes')
    parser.add_argument('--model_name', type=str, default='autove')
    parser.add_argument('--exp_id', type=int, default=1111)
    args = parser.parse_args()


    # new argparser
    model_name = args.model_name
    new_args = getattr(ap, model_name)(ap.parser)
    new_args.model_name = model_name


    # modify -----------------
    # new_args.is_train_dis = 0
    # new_args.gan_prior_prob = 0
    # new_args.social_pooling_type = 0
    new_args.gamma = 0
    new_args.is_train_w_nuscenes = 0
    # ------------------------


    # old argparser
    folder_name = args.dataset_type + '_' + args.model_name + '_model' + str(args.exp_id)
    save_dir = os.path.join('./saved_models/', folder_name)
    if save_dir != '' and not os.path.exists(save_dir):
        sys.exit('[Error] no such files or directories {%s}' % save_dir)

    # load pre-defined training settings or save current settings
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        old_args = pickle.load(f)

    # assign
    for old_arg in vars(old_args):
        setattr(new_args, old_arg, getattr(old_args, old_arg))
        # print(old_arg, getattr(old_args, old_arg))

    # check
    for new_arg in vars(new_args):
        new_value = getattr(new_args, new_arg)
        try:
            old_value = getattr(old_args, new_arg)
            if (new_value is not old_value):
                print(">> [False Assignment] variable name :", new_arg, ', value :', new_value)
                sys.exit()
        except:
            print(">> [New] variable name :", new_arg, ', value :', new_value)




    with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(new_args, f)






if __name__ == '__main__':
    main()

