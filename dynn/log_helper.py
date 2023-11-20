import numpy as np
import mlflow
import os
import torch
import scipy
from grit.utils import free, get_abs_path

from grit.utils import free
from dynn.calibration import calibration_curve

def get_display(key, cum_metric):
    if 'correct' in key:
            return  100*np.mean(cum_metric)
    else:
        return np.mean(cum_metric)

def we_want_to_see_it(metric_key):
    if 'pow' in metric_key:
        return False
    if 'list' in metric_key:
        return False
    if 'score' in metric_key:
        return False
    if 'cheating' in metric_key:
        return False
    return True

# here we change the name of the metric to be displayed
def key_to_name_display(metric_key):
    metric_name = metric_key.replace('correct', 'acc')
    metric_name = metric_name.replace('count_', '')
    metric_name = metric_name.replace('count', '')
    metric_name = metric_name.replace('per_gate', '')
    if metric_name[-1] == '_':
        metric_name = metric_name[:-1]
    return metric_name


def log_aggregate_metrics_mlflow(prefix_logger, metrics_dict, gates_count):
    log_dict = {}
    for metric_key, val in metrics_dict.items():
        if we_want_to_see_it(metric_key):
            metric_name_display = key_to_name_display(metric_key)
            cumul_metric, total = val
            if total > 0:
                if type(cumul_metric) is list: 
                    if (len(cumul_metric) == gates_count or len(cumul_metric) == gates_count+1) and 'per_gate' in metric_key:# if the length is the number of gates we want to see all of them
                        for g, cumul_metric_per_gate in enumerate(cumul_metric):
                            log_dict[prefix_logger+'/'+metric_name_display+ str(g)]  = get_display(metric_key, cumul_metric_per_gate)/total
                    else:
                        log_dict[prefix_logger+'/'+metric_name_display]  = get_display(metric_key, np.mean(cumul_metric))/total
                elif type(cumul_metric) is dict :
                    for g, cumul_metric_per_gate in cumul_metric.items():
                            log_dict[prefix_logger+'/'+metric_name_display+ str(g)]  = get_display(metric_key, cumul_metric_per_gate)/total
                else:
                    log_dict[prefix_logger+'/'+metric_name_display] = get_display(metric_key, cumul_metric)/total
    return log_dict

def aggregate_metrics(metrics_to_aggregate_dict, metrics_dict, gates_count):
    # automate what we do for each type of entries in the dict.
    for metric_key, metric_total_tuple in metrics_to_aggregate_dict.items():
        if metric_key not in metrics_dict:
            metrics_dict[metric_key] = metric_total_tuple
        else: # add the new value based on the type of the variable
            metric = metric_total_tuple[0]
            batch_size = metric_total_tuple[1]
            total = metrics_dict[metric_key][1] + batch_size
            if type(metric) is list:
                if (len(metric) == gates_count or len(metric) == gates_count+1)  and 'per_gate' in metric_key: # we maintain
                    aggregated_metric= []
                    for g, per_gate_metric in enumerate(metric):
                        aggregated_metric.append(metrics_dict[metric_key][0][g] + per_gate_metric)
                else: # we just concat
                    aggregated_metric = metrics_dict[metric_key][0] + metric
            elif type(metric) is dict:
                if (len(metric) == gates_count or len(metric) == gates_count+1)  and 'gate' in metric_key: # we maintain
                    aggregated_metric= {k: metric.get(k, 0) + metrics_dict[metric_key][0].get(k, 0) for k in set(metric) | set(metrics_dict[metric_key][0])}

                else: # we just concat
                    print('Warning, I dont know how to aggregate',metric)
            else:
                aggregated_metric = metrics_dict[metric_key][0] + metric
            metrics_dict[metric_key] = (aggregated_metric, total)
    return metrics_dict

def process_things(things_of_interest, gates_count, targets, batch_size, cost_per_exit, custom_metrics = False):
    """
    This function transforms the model outputs to various metrics. The metrics have the format (count, batch_size) to be aggregated later with other metrics.
    """

    metrics_to_aggregate_dict = {} # each entry has the format = (value, batch_size)
    if 'final_logits' in things_of_interest:
        final_y_logits = things_of_interest['final_logits']
        _, pred_final_head = final_y_logits.max(1)
        metrics_to_aggregate_dict['correct'+str(gates_count)] = (pred_final_head.eq(targets).sum().item(), batch_size)

        # uncertainty related stats to be aggregated
        p_max, entropy, average_ece, margins, entropy_pow = compute_detached_uncertainty_metrics(final_y_logits, targets)
        score = compute_detached_score(final_y_logits, targets)
        metrics_to_aggregate_dict['final_p_max'] = (p_max, batch_size)
        metrics_to_aggregate_dict['final_entropy'] = (entropy, batch_size)
        metrics_to_aggregate_dict['final_pow_entropy'] = (entropy_pow, batch_size)
        metrics_to_aggregate_dict['final_margins'] = (margins, batch_size)
        metrics_to_aggregate_dict['final_ece'] = (average_ece*batch_size*100.0, batch_size)
        metrics_to_aggregate_dict['all_final_score'] = (score, batch_size)
        if 'sample_exit_level_map' in things_of_interest:
            score_filtered = np.array(score)[free(things_of_interest['sample_exit_level_map'] == gates_count)]
            metrics_to_aggregate_dict['score_per_final_gate'] = (list(score_filtered), batch_size)
    if 'intermediate_logits' in things_of_interest:
        intermediate_logits = things_of_interest['intermediate_logits']

        # how much we could get with perfect gating

        shape_of_correct = pred_final_head.eq(targets).shape
        correct_class_cheating = torch.full(shape_of_correct,False).to(pred_final_head.device)

        entries = ['ens_correct_per_gate','correct_per_gate', 'correct_cheating_per_gate','list_correct_per_gate','margins_per_gate',
                   'p_max_per_gate','entropy_per_gate','pow_entropy_per_gate','ece_per_gate','score_per_gate', 'all_score_per_gate', 'wacc_per_gate']
        for entry in entries:
            metrics_to_aggregate_dict[entry] = ([0 for _ in range(gates_count)], batch_size)
        for g in range(gates_count):

            # normal accuracy
            _, predicted_inter = intermediate_logits[g].max(1)
            correct_gate = predicted_inter.eq(targets)
            metrics_to_aggregate_dict['correct_per_gate'][0][g] = correct_gate.sum().item()
            # ensembling
            logits_up_to_g = torch.cat([intermediate_logits[i][:,:,None] for i in range(g+1)], dim=2)
            if 'p_exit_at_gate' in things_of_interest: # we compute the ensembling with weighted by prob of exit
                logits_up_to_g = torch.permute(logits_up_to_g, (1, 0, 2))
                weighted_logits_up_to_g = logits_up_to_g * things_of_interest['p_exit_at_gate'][:,:g+1]
                weighted_logits_up_to_g = torch.permute(weighted_logits_up_to_g, (1,0, 2))
                ens_logits = torch.mean(weighted_logits_up_to_g, dim=2)
            else:
                ens_logits = torch.mean(logits_up_to_g, dim=2)

            _, ens_predicted_inter = ens_logits.max(1)
            ens_correct_gate = ens_predicted_inter.eq(targets)
            # metrics_to_aggregate_dict['ens_correct_per_gate'][0][g] = ens_correct_gate.sum().item()


            # keeping all the corrects we have from previous gates
            #correct_class_cheating += correct_gate
            # metrics_to_aggregate_dict['correct_cheating_per_gate'][0][
            #     g] = correct_class_cheating.sum().item() # getting all the corrects we can

            p_max, entropy, average_ece, margins, entropy_pow = compute_detached_uncertainty_metrics(
                intermediate_logits[g], targets)
            score = compute_detached_score(intermediate_logits[g], targets)
            # metrics_to_aggregate_dict['list_correct_per_gate'][0][g] = list(free(correct_gate))
            metrics_to_aggregate_dict['margins_per_gate'][0][g] = margins
            metrics_to_aggregate_dict['p_max_per_gate'][0][g] = p_max
            metrics_to_aggregate_dict['entropy_per_gate'][0][g] = entropy
            #metrics_to_aggregate_dict['pow_entropy_per_gate'][0][g] = entropy_pow
            metrics_to_aggregate_dict['ece_per_gate'][0][g] = 100.0*average_ece*batch_size
            if 'sample_exit_level_map' in things_of_interest:
                score_filtered = np.array(score)[free(things_of_interest['sample_exit_level_map'] == g)]
                metrics_to_aggregate_dict['score_per_gate'][0][g] = list(score_filtered)

            metrics_to_aggregate_dict['all_score_per_gate'][0][g] = score

        correct_class_cheating += pred_final_head.eq(targets)  # getting all the corrects we can
        metrics_to_aggregate_dict['cheating_correct'] = (correct_class_cheating.sum().item(), batch_size)

    # if 'sets_general' in things_of_interest: #
    #
    #     keys_sets = ['sets_general','sets_gated','sets_gated_all','sets_gated_strict'  ]
    #     for type_of_sets in keys_sets: #we use different strategies to build the conformal sets.
    #         conf_sets_dict = things_of_interest[type_of_sets]
    #         for alpha, conf_sets  in conf_sets_dict.items():
    #             # C, emp_alpha = compute_coverage_and_inef(conf_sets, targets)
    #             summed_C = C * batch_size
    #             summed_alpha = emp_alpha * batch_size
    #
    #             metrics_to_aggregate_dict[type_of_sets+'_C_'+str(alpha)] = (summed_C, batch_size)
    #             metrics_to_aggregate_dict[type_of_sets+'_emp_alpha_'+str(alpha)] = (summed_alpha, batch_size)



    if 'gated_y_logits' in things_of_interest:
        gated_y_logits = things_of_interest['gated_y_logits']
        # _, _ , average_ece ,_ ,_ = compute_detached_uncertainty_metrics(gated_y_logits, targets)
        score = compute_detached_score(gated_y_logits, targets)
        _, predicted = gated_y_logits.max(1)
        metrics_to_aggregate_dict['gated_correct_count'] = (predicted.eq(targets).sum().item(), batch_size)
        # metrics_to_aggregate_dict['gated_ece_count'] = (100.0*average_ece*batch_size, batch_size)
        metrics_to_aggregate_dict['gated_score'] = (score, batch_size)

    if 'num_exits_per_gate' in things_of_interest:
        num_exits_per_gate = things_of_interest['num_exits_per_gate']
        gated_y_logits = things_of_interest['gated_y_logits']
        _, predicted = gated_y_logits.max(1)
        total_cost = compute_cost(num_exits_per_gate, cost_per_exit)
        metrics_to_aggregate_dict['total_cost'] = (total_cost, batch_size)
    if 'sample_exit_level_map' in things_of_interest:

        correct_number_per_gate_batch = compute_correct_number_per_gate(
            gates_count,
            things_of_interest['sample_exit_level_map'],
            targets,
            predicted)

        metrics_to_aggregate_dict['percent_exit_per_gate'] = ([0 for _ in range(gates_count+1)], batch_size) # +1 because we count the last gate as well.
        for g, pred_tuple in correct_number_per_gate_batch.items():
            metrics_to_aggregate_dict['gated_correct_count_'+str(g)]= (pred_tuple[0], pred_tuple[1])
            metrics_to_aggregate_dict['percent_exit_per_gate'][0][g] = pred_tuple[1]

    # GATE associated metrics
    if 'exit_count_optimal_gate' in things_of_interest:
        exit_count_optimal_gate = things_of_interest['exit_count_optimal_gate']
        correct_exit_count = things_of_interest['correct_exit_count']
        metrics_to_aggregate_dict['exit_count_optimal_gate'] = (exit_count_optimal_gate, batch_size)
        metrics_to_aggregate_dict['correct_exit_count'] = (correct_exit_count, batch_size * gates_count) # the correct count is over all gates

    return metrics_to_aggregate_dict

def setup_mlflow(run_name: str, cfg, experiment_name):
    print(run_name)
    project = experiment_name
    mlruns_path = get_abs_path(["mlruns"])
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(project)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(vars(cfg))



def compute_detached_uncertainty_metrics(logits, targets, is_binary_task = False):
    # if logits.shape
    if logits.shape[1] == 1: # binary task
        prob_out = torch.sigmoid(logits)
        complement = 1 - prob_out
        probs = torch.squeeze(torch.stack([prob_out, complement], dim = 1))
    else:
        probs = torch.nn.functional.softmax(logits, dim=1)

    top2prob, _ = torch.topk(probs, 2)

    p_max = top2prob[:,0]
    next_p_max = top2prob[:,1]
    margins = p_max-next_p_max

    p_max = free(p_max)
    margins = free(margins)

    entropy = scipy.stats.entropy(free(probs), axis=1)

    pow_probs = probs**2
    pow_probs = pow_probs / pow_probs.sum(dim=1, keepdim=True)
    entropy_pow = scipy.stats.entropy(free(pow_probs), axis=1)

    ece = -1
    if targets is not None:
        _, predicted = logits.max(1)
        correct = predicted.eq(targets)
        ground_truth = free(correct)
        _, _, ece = calibration_curve(ground_truth, p_max,strategy='quantile',n_bins=15)


    return list(p_max), list(entropy), ece, list(margins), list(entropy_pow)



def compute_detached_score(logits, targets, is_binary_task = False): # score value to obtain conformal intervals
    if logits.shape[1] == 1:
        prob_out = torch.sigmoid(logits)
        complement = 1 - prob_out
        probs = torch.squeeze(torch.stack([prob_out, complement], dim = 1))
    else:
        probs = torch.nn.functional.softmax(logits, dim=1)
    s = 1-probs[np.arange(logits.shape[0]), free(targets)]
    return list(free(s))

def compute_cost(num_exits_per_gate, cost_per_exit):
    cost_per_gate = [
        free(num) * cost_per_exit[g]
        for g, num in enumerate(num_exits_per_gate)
    ]
    # the last cost_per gate should be equal to the last num
    return np.sum(cost_per_gate)

def compute_correct_number_per_gate(number_of_gates: int,
                                    sample_exit_level_map: torch.Tensor,
                                    targets: torch.Tensor,
                                    predicted: torch.Tensor
                                    ):
    """
    Computes the number of correct predictions a gate made only on the samples that it exited.

    :param number_of_gates Number of gates in the dynn. We need this in case some gates are never reached
    :param sample_exit_level_map: A tensor the same size as targets that holds the exit level for each sample
    :param targets: ground truths
    :param predicted: predictions of the dynamic network
    :return: A map  where the key is gate_idx and the value is a tuple (correct_count, total_predictions_of_gate_count)
    """
    result_map = {}
    for gate_idx in range(number_of_gates+1):
        gate_predictions_idx = (sample_exit_level_map == gate_idx).nonzero()
        pred_count = len(gate_predictions_idx)
        correct_pred_count = torch.sum((predicted[gate_predictions_idx].eq(targets[gate_predictions_idx]))).item()
        result_map[gate_idx] = (correct_pred_count, pred_count)
    return result_map


