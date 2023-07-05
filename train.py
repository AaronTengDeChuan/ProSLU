import random
import numpy as np

from models.model import ModelManager
from utils.config import *
from utils.loader import DatasetManager
from utils.process import Processor

if __name__ == "__main__":
    if args.fix_seed:
        # Fix the random seed of package random.
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

        # Fix the random seed of Pytorch when using CPU.
        torch.manual_seed(args.random_seed)
        torch.random.manual_seed(args.random_seed)

        # Fix the random seed of Pytorch when using GPU.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
            torch.cuda.manual_seed(args.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if not args.do_eval:
        # Instantiate a dataset object.
        dataset = DatasetManager(args)
        dataset.quick_build()

        # Instantiate a network model object.
        model = ModelManager(
            args, len(dataset.word_alphabet),
            len(dataset.slot_alphabet),
            len(dataset.intent_alphabet))

        # To train and evaluate the models.
        process = Processor(dataset, model, args)
        process.train()

        test_result = process.validate(
            os.path.join(args.save_dir, "model/model.pkl"),
            os.path.join(args.save_dir, "model/dataset.pkl")
        )
        mylogger.info('\nAccepted performance: ' +
                      " ; ".join([f"{k}={v}" for k, v in test_result.items()]) +
                      " at test dataset;\n")
    else:
        # for only evaluation
        ## python train.py -de -ld "save/BERTSLU++/True_ELECTRA_8_0.4_0.0008_4e-05_64_128_2021-12-20\ 032603/model" -g -fs -es -uf -up -ui -mt ELECTRA -bs 8 -lr 0.0008 -blr 4e-05
        assert args.load_dir is not None
        # load dataset
        dataset_path = os.path.join(args.load_dir, "dataset.pkl")
        if args.gpu:
            dataset = torch.load(dataset_path)
        else:
            dataset = torch.load(dataset_path, map_location=torch.device('cpu'))

        # Instantiate a network model object.
        model = ModelManager(
            args, len(dataset.word_alphabet),
            len(dataset.slot_alphabet),
            len(dataset.intent_alphabet))

        args = dataset.args
        if "PretrainModel/bert" in args.model_type_path:
            args.model_type_path = f"./{args.model_type_path}"
        # load model during 'Processor.init'
        process = Processor(dataset, model, args)

        # load model during 'process.validate'
        model_path = os.path.join(args.load_dir, "model.pkl")
        mylogger.info('\nAccepted performance: ' +
                      " ; ".join([f"{k}={v}" for k, v in process.validate(model_path, None).items()]) +
                      " at test dataset;\n")
