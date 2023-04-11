from utils.libraries import *
import argumentparser as ap

def main():

    # parent of working directory is base
    abspath = os.path.dirname(os.path.realpath(__file__))
    os.chdir(Path(abspath).parent.absolute())

    exp_id = 1023
    model_name = 'autove'
    dataset_type = 'ETRI'


    # old argparser
    folder_name = dataset_type + '_' + model_name + '_model' + str(exp_id)
    save_dir = os.path.join('./saved_models/', folder_name)
    if save_dir != '' and not os.path.exists(save_dir):
        sys.exit('[Error] no such files or directories {%s}' % save_dir)

    # load pre-defined training settings or save current settings
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        old_args = pickle.load(f)

    # assign
    for old_arg in vars(old_args):
        old_value = getattr(old_args, old_arg)
        print(">> variable name :", old_arg, ', value :', old_value)

if __name__ == '__main__':
    main()

