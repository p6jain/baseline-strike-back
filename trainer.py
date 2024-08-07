import numpy
import time
import evaluate
import torch
import kb
import utils
import os
import random
from tensorboardX import SummaryWriter
import pdb


def log_eval_scores(writer, valid_score, test_score, num_iter):
    for metric in ['mrr','hits10','hits1']:
        writer.add_scalar('{}/valid_m'.format(metric), valid_score['m'][metric] , num_iter)
        writer.add_scalar('{}/valid_e1'.format(metric), valid_score['e1'][metric] , num_iter)
        writer.add_scalar('{}/valid_e2'.format(metric), valid_score['e2'][metric] , num_iter)

        writer.add_scalar('{}/test_m'.format(metric), test_score['m'][metric] , num_iter)
        writer.add_scalar('{}/test_e1'.format(metric), test_score['e1'][metric] , num_iter)
        writer.add_scalar('{}/test_e2'.format(metric), test_score['e2'][metric] , num_iter)
    return


class Trainer(object):
    def __init__(self, scoring_function, regularizer, loss, optim, train, valid, test, verbose=0, batch_size=1000,
                 hooks=None , eval_batch=100, negative_count=10, gradient_clip=None, regularization_coefficient=0.01,
                 save_dir="./logs", scheduler=None, arguments_str=''):
        super(Trainer, self).__init__()
        self.scoring_function = scoring_function
        self.loss = loss
        self.regularizer = regularizer
        self.train = train
        self.test = test
        self.valid = valid
        self.optim = optim
        self.batch_size = batch_size
        self.negative_count = negative_count
        self.ranker = evaluate.ranker(self.scoring_function, kb.union([train.kb, valid.kb, test.kb]))
        self.eval_batch = eval_batch
        self.gradient_clip = gradient_clip
        self.regularization_coefficient = regularization_coefficient
        print("Using Regularizing coef", self.regularization_coefficient)
        self.save_directory = save_dir
        self.best_mrr_on_valid = {"valid_m":{"mrr":0.0}, "test_m":{"mrr":0.0},
                                          "valid_e2":{"mrr":0.0}, "test_e2":{"mrr":0.0},
                                          "valid_e1":{"mrr":0.0}, "test_e1":{"mrr":0.0}}
        self.verbose = verbose
        self.hooks = hooks if hooks else []
        self.scheduler = scheduler

        try:
            self.flag_add_reverse = self.scoring_function.flag_add_reverse
        except:
            self.flag_add_reverse = 0

        self.arguments_str = arguments_str
    
    def step(self):
        #'''
        ##use all ent as neg sample
        flag_using_full_softmax = 0

        if self.negative_count == 0 or self.loss.name == 'crossentropy_loss_AllNeg_subsample':  # use all ent as neg sample
            ns = None
            no = None
            s, r, o, _, _ = self.train.tensor_sample(self.batch_size, 1)
            flag_using_full_softmax = 1
        else:
            s, r, o, ns, no = self.train.tensor_sample(self.batch_size, self.negative_count)

        '''
        try:
            assert ns.shape[1] == 2
        except:
            print(ns.shape,no.shape)
            exit()
        '''
        using_sm = 1
        if flag_using_full_softmax:
            using_sm = 0

        flag = random.randint(1,10001)
        if flag>9950:
            flag_debug = 1
        else:
            flag_debug = 0

        if flag_debug:
            fp  = self.scoring_function(s, r, o, flag_debug=flag_debug+1)#using_sm=using_sm, flag_debug=flag_debug+1)
            fno = self.scoring_function(s, r, no, flag_debug=flag_debug+1)#using_sm=using_sm, flag_debug=flag_debug+1)
            fns = self.scoring_function(ns, r, o, flag_debug=flag_debug+1)#using_sm=using_sm, flag_debug=flag_debug+1)
        else:
            fp  = self.scoring_function(s, r, o, flag_debug=flag_debug+1)#using_sm=using_sm, flag_debug=0)
            fno = self.scoring_function(s, r, no, flag_debug=flag_debug+1)#using_sm=using_sm, flag_debug=0)
            fns = self.scoring_function(ns, r, o, flag_debug=flag_debug+1)#using_sm=using_sm, flag_debug=0)

        #pdb.set_trace()

        '''
        fp = self.scoring_function(s, r, o)
        fns = self.scoring_function(ns, r, o)
        fno = self.scoring_function(s, r, no)
        '''

        if self.regularization_coefficient is not None:
            #'''
            reg = self.regularizer(s, r, o)#, reg_val=3) #+ self.regularizer(ns, r, o) + self.regularizer(s, r, no)
            if self.scoring_function.reg != 3:
                reg = reg/self.batch_size#/(self.batch_size*self.scoring_function.embedding_dim)
                ##dividing by dim size is a bit too much!!
            #'''
            '''
            reg = self.regularizer(s, r, o) + self.regularizer(ns, r, o) + self.regularizer(s, r, no)
            reg = reg/(self.batch_size*self.scoring_function.embedding_dim*(1+2*self.negative_count))
            '''
        else:
            reg = 0
        #'''
        ##use all ent as neg sample
        if flag_using_full_softmax:
            if self.loss.name == 'crossentropy_loss_AllNeg_subsample':
                loss = self.loss(s,fns, self.negative_count) + self.loss(o, fno, self.negative_count) + self.regularization_coefficient*reg
            else:
                #loss = self.loss(s, fns) + self.loss(o, fno) + self.regularization_coefficient*reg
                if self.flag_add_reverse==0 or self.scoring_function.flag_avg_scores:
                    loss = self.loss(s, fns) + self.loss(o, fno) + self.regularization_coefficient*reg 
                else:
                    loss = self.loss(o, fno) + self.regularization_coefficient*reg

        #'''
        #'''
        else:
            #pdb.set_trace()
            #loss = self.loss(fp, fns) + self.regularization_coefficient*reg
            a = self.loss(fp, fns) 
            b = self.loss(fp, fno) 
            c = self.regularization_coefficient*reg
            #loss = self.loss(fp, fns) + self.loss(fp, fno) + self.regularization_coefficient*reg
            loss = a+b+c
        #'''
       
        #if flag_debug:
        #    print("s:",a,"  o: ",b,"  reg: ",c)
 
        x = loss.item()
        rg = reg.item()
        self.optim.zero_grad()
        loss.backward()
        if(self.gradient_clip is not None):
            torch.nn.utils.clip_grad_norm(self.scoring_function.parameters(), self.gradient_clip)
        self.optim.step()
        debug = ""
        if "post_epoch" in dir(self.scoring_function):
            debug = self.scoring_function.post_epoch()
        return x, rg, debug

    def save_state(self, mini_batches, valid_score, test_score):
        state = dict()
        state['mini_batches'] = mini_batches
        state['epoch'] = mini_batches*self.batch_size/self.train.kb.facts.shape[0]
        state['model_name'] = type(self.scoring_function).__name__
        state['model_weights'] = self.scoring_function.state_dict()
        state['optimizer_state'] = self.optim.state_dict()
        state['optimizer_name'] = type(self.optim).__name__
        state['valid_score_e2'] = valid_score['e2']
        state['test_score_e2'] = test_score['e2']
        state['valid_score_e1'] = valid_score['e1']
        state['test_score_e1'] = test_score['e1']
        state['valid_score_m'] = valid_score['m']
        state['test_score_m'] = test_score['m']

        
        state['entity_map'] = self.train.kb.entity_map
        state['reverse_entity_map'] = self.train.kb.reverse_entity_map
        state['relation_map'] = self.train.kb.relation_map
        state['reverse_relation_map'] = self.train.kb.reverse_relation_map

        # state['additional_params'] = self.train.kb.additional_params
        state['nonoov_entity_count'] = self.train.kb.nonoov_entity_count


        filename = os.path.join(self.save_directory, "epoch_%.1f_val_%5.2f_%5.2f_%5.2f_test_%5.2f_%5.2f_%5.2f.pt"%(state['epoch'],
                                                                                           state['valid_score_e2']['mrr'], 
                                                                                           state['valid_score_e1']['mrr'], 
                                                                                           state['valid_score_m']['mrr'],
                                                                                           state['test_score_e2']['mrr'],
                                                                                           state['test_score_e1']['mrr'],
                                                                                           state['test_score_m']['mrr']))


        #torch.save(state, filename)
        try:
            if(state['valid_score_m']['mrr'] >= self.best_mrr_on_valid["valid_m"]["mrr"]):
                print(self.arguments_str)
                print("Best Model details:\n","valid_m",str(state['valid_score_m']), "test_m",str(state["test_score_m"]),
                                          "valid",str(state['valid_score_e2']), "test",str(state["test_score_e2"]),
                                          "valid_e1",str(state['valid_score_e1']),"test_e1",str(state["test_score_e1"]))
                best_name = os.path.join(self.save_directory, "best_valid_model.pt")
                self.best_mrr_on_valid = {"valid_m":state['valid_score_m'], "test_m":state["test_score_m"], 
                                          "valid":state['valid_score_e2'], "test":state["test_score_e2"],
                                          "valid_e1":state['valid_score_e1'], "test_e1":state["test_score_e1"]}

                if(os.path.exists(best_name)):
                    os.remove(best_name)
                torch.save(state, best_name)#os.symlink(os.path.realpath(filename), best_name)
        except:
            utils.colored_print("red", "unable to save model")

    def load_state(self, state_file):
        state = torch.load(state_file)
        if state['model_name'] != type(self.scoring_function).__name__:
            utils.colored_print('yellow', 'model name in saved file %s is different from the name of current model %s' %
                                (state['model_name'], type(self.scoring_function).__name__))
        self.scoring_function.load_state_dict(state['model_weights'])
        if state['optimizer_name'] != type(self.optim).__name__:
            utils.colored_print('yellow', ('optimizer name in saved file %s is different from the name of current '+
                                          'optimizer %s') %
                                (state['optimizer_name'], type(self.optim).__name__))
        self.optim.load_state_dict(state['optimizer_state'])
        return state['mini_batches']

     


    def start(self, steps=50, batch_count=(20, 10), mb_start=0, logs_dir=""):
        start = time.time()
        losses = []
        count = 0
        
        #CPU
        # self.scoring_function=self.scoring_function.cpu()
        '''
        self.scoring_function.eval()
        valid_score = evaluate.evaluate("valid", self.ranker, self.valid.kb, self.eval_batch,
                                        verbose=self.verbose, hooks=self.hooks)
        test_score = evaluate.evaluate("test ", self.ranker, self.test.kb, self.eval_batch,
                                           verbose=self.verbose, hooks=self.hooks)
        self.scoring_function.train()
        '''
        writer = SummaryWriter(logs_dir)

        print("Starting training")
        #exit()
        
        for i in range(mb_start, steps):
            l, reg, debug = self.step()
            writer.add_scalar('loss/batch_loss', l, i)

            losses.append(l)
            suffix = ("| Current Loss %8.4f | "%l) if len(losses) != batch_count[0] else "| Average Loss %8.4f | " % \
                                                                                         (numpy.mean(losses))
            suffix += "reg %6.3f | time %6.0f ||"%(reg, time.time()-start)
            suffix += debug
            prefix = "Mini Batches %5d or %5.1f epochs"%(i+1, i*self.batch_size/self.train.kb.facts.shape[0])
            utils.print_progress_bar(len(losses), batch_count[0],prefix=prefix, suffix=suffix)
            if len(losses) >= batch_count[0]:
                count += 1
                writer.add_scalar('loss/avg_loss', numpy.mean(losses) , i)
                
                losses = []
                if count == batch_count[1]:
                    self.scoring_function.eval()
                    valid_score = evaluate.evaluate("valid", self.ranker, self.valid.kb, self.eval_batch,
                                                    verbose=self.verbose, hooks=self.hooks)
                    test_score = evaluate.evaluate("test ", self.ranker, self.test.kb, self.eval_batch,
                                                   verbose=self.verbose, hooks=self.hooks)
                    
                    log_eval_scores(writer, valid_score,test_score,i)

                    self.scoring_function.train()
                    self.scheduler.step(valid_score['m']['mrr']) #Scheduler to manage learning rate added
                    count = 0
                    print()
                    self.save_state(i, valid_score, test_score)
        print()
        print("Ending")
        #print(self.best_mrr_on_valid["valid_m"])
        #print(self.best_mrr_on_valid["test_m"])
        print(self.best_mrr_on_valid)

