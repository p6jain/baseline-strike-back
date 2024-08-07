import kb
import data_loader
import trainer
import torch
import losses
import models
import argparse
import os
import datetime
import json
import utils
import extra_utils
import sys
import torch.optim.lr_scheduler as lr_scheduler
import re 
import pdb

import numpy as np
import torch 

# set random seeds
torch.manual_seed(32) 
np.random.seed(12)


has_cuda = torch.cuda.is_available()
if not has_cuda:
    utils.colored_print("yellow", "CUDA is not available, using cpu")


def main(dataset_root, save_dir, tflogs_dir, model_name, model_arguments, loss_function, learning_rate, batch_size,
         regularization_coefficient, regularizer, gradient_clip, optimizer_name, max_epochs, negative_sample_count, hooks,
         eval_every_x_mini_batches, eval_batch_size, resume_from_save, introduce_oov, verbose, batch_norm, flag_add_reverse, avg_scores, arguments_str):
    ktrain = kb.kb(os.path.join(dataset_root, 'train.txt'))
    if introduce_oov:
        if not "<OOV>" in ktrain.entity_map.keys():
            ktrain.entity_map["<OOV>"] = len(ktrain.entity_map)
            ktrain.nonoov_entity_count = ktrain.entity_map["<OOV>"]+1
    ktest = kb.kb(os.path.join(dataset_root, 'test.txt'), em=ktrain.entity_map, rm=ktrain.relation_map,
                  add_unknowns=not introduce_oov, nonoov_entity_count=ktrain.nonoov_entity_count)
    kvalid = kb.kb(os.path.join(dataset_root, 'valid.txt'), em=ktrain.entity_map, rm=ktrain.relation_map,
                   add_unknowns=not introduce_oov, nonoov_entity_count=ktrain.nonoov_entity_count)
    if(verbose > 0):
        utils.colored_print("yellow", "VERBOSE ANALYSIS only for FB15K")
        tpm = extra_utils.type_map_fine(dataset_root)#extra_utils.fb15k_type_map_fine()
        ktrain.augment_type_information(tpm)
        ktest.augment_type_information(tpm)
        kvalid.augment_type_information(tpm)
        hooks = extra_utils.load_hooks(hooks, ktrain)

    #protocol
    '''
    if re.search("_lx",model_name):#model_name == "typed_model_lx":
        assert flag_add_reverse == 1    
    else:
        assert flag_add_reverse == 0

    if flag_add_reverse == 1:
        assert re.search("_lx",model_name)
    '''
    #protocol
    print("Reg1/2 now normalised")
    if introduce_oov:
        first_zero_val = True
    else:
        first_zero_val = False

    dltrain = data_loader.data_loader(ktrain, has_cuda, loss=loss_function, flag_add_reverse=flag_add_reverse, first_zero=first_zero_val)
    dlvalid = data_loader.data_loader(kvalid, has_cuda, loss=loss_function, first_zero=first_zero_val)#, flag_add_reverse=flag_add_reverse)
    dltest = data_loader.data_loader(ktest, has_cuda, loss=loss_function, first_zero=first_zero_val)#, flag_add_reverse=flag_add_reverse)

    model_arguments['entity_count'] = len(ktrain.entity_map)
    if regularizer:
        print("Using reg ", regularizer)
        model_arguments['reg'] = regularizer

    if flag_add_reverse:
        model_arguments['relation_count'] = len(ktrain.relation_map)*2
        model_arguments['flag_add_reverse'] = flag_add_reverse
        model_arguments['flag_avg_scores'] = avg_scores
        if 0:
            print("Hard code alert!!: Using reg 3")
            model_arguments['reg'] = 3
    else:
        model_arguments['relation_count'] = len(ktrain.relation_map)

    model_arguments['batch_norm'] = batch_norm


    print("model_arguments", model_arguments)
    scoring_function = getattr(models, model_name)(**model_arguments)
    if has_cuda:
        scoring_function = scoring_function.cuda()
    loss = getattr(losses, loss_function)()
    optim = getattr(torch.optim, optimizer_name)(scoring_function.parameters(), lr=learning_rate)#, weight_decay= 0,  initial_accumulator_value= 0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optim, 'max', patience = 2, verbose=True)#mrr tracking 

    if(not eval_batch_size):
        eval_batch_size = max(50, batch_size*2*negative_sample_count//len(ktrain.entity_map))
    tr = trainer.Trainer(scoring_function, scoring_function.regularizer, loss, optim, dltrain, dlvalid, dltest,
                         batch_size=batch_size, eval_batch=eval_batch_size, negative_count=negative_sample_count,
                         save_dir=save_dir, gradient_clip=gradient_clip, hooks=hooks,
                         regularization_coefficient=regularization_coefficient, verbose=verbose, scheduler=scheduler, arguments_str=arguments_str)#0.01)

    if resume_from_save:
        mb_start = tr.load_state(resume_from_save)
    else:
        mb_start = 0
    max_mini_batch_count = int(max_epochs*ktrain.facts.shape[0]/batch_size)
    print("max_mini_batch_count: %d, eval_batch_size %d" % (max_mini_batch_count, eval_batch_size))
    tr.start(max_mini_batch_count, [eval_every_x_mini_batches//20, 20], mb_start, tflogs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help="Name of the dataset as in data folder", required=True)
    parser.add_argument('-m', '--model', help="model name as in models.py", required=True)
    parser.add_argument('-a', '--model_arguments', help="model arguments as in __init__ of "
                                                        "model (Excluding entity and relation count) "
                                                        "This is a json string", required=True)
    parser.add_argument('-o', '--optimizer', required=False, default='Adagrad')
    parser.add_argument('-l', '--loss', help="loss function name as in losses.py", required=True)
    parser.add_argument('-r', '--learning_rate', required=True, type=float)
    parser.add_argument('-g', '--regularization_coefficient', required=True, type=float)
    parser.add_argument('-g_reg', '--regularizer', required=True, default=0.0, type=float)
    parser.add_argument('-c', '--gradient_clip', required=False, type=float)
    parser.add_argument('-e', '--max_epochs', required=False, type=int, default=1000)
    parser.add_argument('-b', '--batch_size', required=False, type=int, default=2000)
    parser.add_argument('-x', '--eval_every_x_mini_batches', required=False, type=int, default=2000)
    parser.add_argument('-y', '--eval_batch_size', required=False, type=int, default=0)
    parser.add_argument('-n', '--negative_sample_count', required=False, type=int, default=200)
    parser.add_argument('-s', '--save_dir', required=False)
    parser.add_argument('-u', '--resume_from_save', required=False)
    parser.add_argument('-v', '--oov_entity', required=False)
    parser.add_argument('-q', '--verbose', required=False, default=0, type=int)
    parser.add_argument('-z', '--debug', required=False, default=0, type=int)
    parser.add_argument('-k', '--hooks', required=False, default="[]")
    parser.add_argument('-bn', '--batch_norm', required=False, default=0, type=int)
    parser.add_argument('-msg', '--message', required=False)
    parser.add_argument('--data_repository_root', required=False, default='data')

    parser.add_argument('-inv', '--inverse', required=False, type=int, default=0)
    parser.add_argument('-avg', '--avg_scores', required=False, type=int, default=0)

    #for tensorboard logs
    parser.add_argument('-tf', '--tflogs_dir', required=False)

    arguments = parser.parse_args()
    if arguments.save_dir is None:
        arguments.save_dir = os.path.join("logs", "%s %s %s run on %s starting from %s" % (arguments.model,
                                                                                        arguments.model_arguments,
                                                                                        arguments.loss,
                                                                                        arguments.dataset,
                                                                                     str(datetime.datetime.now())))
    # log_folder = ""#/scratch/cse/phd/csz148211/log_KBI-v2/"
    #log_folder ="/scratch/cse/btech/cs1160329/KBI/"
    log_folder = "/home/prachi/scratch_shakuntala/softmax-exp/"
    #log_folder = "/scratch/cse/phd/csz148211/softmax-exp/"

    arguments.save_dir = log_folder + arguments.save_dir


    arguments.model_arguments = json.loads(arguments.model_arguments)
    arguments.hooks = json.loads(arguments.hooks)
    if not arguments.debug:
        if not os.path.isdir(arguments.save_dir):
            print("Making directory (s) %s" % arguments.save_dir)
            os.makedirs(arguments.save_dir)
        else:
            utils.colored_print("yellow", "directory %s already exists" % arguments.save_dir)
        utils.duplicate_stdout(os.path.join(arguments.save_dir, "log.txt"))

    if arguments.tflogs_dir is None:
        arguments.tflogs_dir=arguments.save_dir
    else:
        arguments.tflogs_dir+=datetime.datetime.now().strftime('_%d-%m-%y_%H.%M.%S')
        if not os.path.isdir(arguments.tflogs_dir):
            print("Making directory (s) %s" % arguments.tflogs_dir)
            os.makedirs(arguments.tflogs_dir)
        else:
            utils.colored_print("yellow", "directory %s already exists" % arguments.tflogs_dir)

    print(arguments)
    print("User Message:: ", arguments.message)
    print("Command:: ", (" ").join(sys.argv))
    dataset_root = os.path.join(arguments.data_repository_root, arguments.dataset)
    main(dataset_root, arguments.save_dir, arguments.tflogs_dir, arguments.model, arguments.model_arguments, arguments.loss,
         arguments.learning_rate, arguments.batch_size, arguments.regularization_coefficient, arguments.regularizer, arguments.gradient_clip,
         arguments.optimizer, arguments.max_epochs, arguments.negative_sample_count, arguments.hooks,
         arguments.eval_every_x_mini_batches, arguments.eval_batch_size, arguments.resume_from_save,
         arguments.oov_entity, arguments.verbose, arguments.batch_norm, arguments.inverse, arguments.avg_scores, arguments.__str__())
