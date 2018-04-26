import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tensorflow.python.keras.layers import Dense, Embedding, TimeDistributed
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.estimator.export.export_output import PredictOutput

X_TRAIN_FILENAME = "X_train.csv.gz"
Y_TRAIN_FILENAME = "Y_train.csv.gz"
SL_TRAIN_FILENAME = "SL_train.csv.gz"
X_VAL_FILENAME = "X_val.csv.gz"
Y_VAL_FILENAME = "Y_val.csv.gz"
SL_VAL_FILENAME = "SL_val.csv.gz"

INPUT_TENSOR_NAME = "ph_inputs"
OUTPUT_TENSOR_NAME = "ph_labels"
SL_TENSOR_NAME = "ph_sequence_lengths"

def train_input_fn(training_dir, hyperparameters):
    
    file_names = {"X": X_TRAIN_FILENAME, "Y": Y_TRAIN_FILENAME, "SL": SL_TRAIN_FILENAME}
    return _input_fn(training_dir, file_names, None, True, hyperparameters)

def eval_input_fn(training_dir, hyperparameters):
    
    file_names = {"X": X_VAL_FILENAME, "Y": Y_VAL_FILENAME, "SL": SL_VAL_FILENAME}
    return _input_fn(training_dir, file_names, 1, False, hyperparameters)

def _input_fn(data_dir, file_names, epochs, do_shuffle, hyperparameters):
    
    xdf = pd.read_csv(os.path.join(data_dir, file_names["X"]), 
                      compression="gzip", header=0, index_col=0, sep=",", dtype=np.float32)
    ydf = pd.read_csv(os.path.join(data_dir, file_names["Y"]), 
                      compression="gzip", header=0, index_col=0, sep=",", dtype=np.float32)
    sldf = pd.read_csv(os.path.join(data_dir, file_names["SL"]), 
                       compression="gzip", header=0, index_col=0, sep=",", dtype=np.float32)

    M = xdf.shape[0]/hyperparameters["MAX_TIMESTEP"]
    X = xdf.as_matrix().reshape(M, hyperparameters["MAX_TIMESTEP"], hyperparameters["INPUT_SIZE"])
    Y = ydf.as_matrix().reshape(M, hyperparameters["MAX_TIMESTEP"], hyperparameters["INPUT_SIZE"])
    SL = sldf.as_matrix().reshape(M,)
    
    return tf.estimator.inputs.numpy_input_fn(
        x = {INPUT_TENSOR_NAME: X, SL_TENSOR_NAME: SL},
        y = Y,
        batch_size = hyperparameters["BATCH_SIZE"],
        num_epochs=epochs,
        shuffle=do_shuffle)()

def serving_input_fn(hyperparameters):
 
    _input = tf.placeholder(tf.float32, [1,  hyperparameters["MAX_TIMESTEP"], hyperparameters["INPUT_SIZE"]], name=INPUT_TENSOR_NAME)
    _index = tf.placeholder(tf.int32, [1, ], name = SL_TENSOR_NAME)
                
    return tf.estimator.export.build_raw_serving_input_receiver_fn({INPUT_TENSOR_NAME: _input, SL_TENSOR_NAME: _index})()

def _calculate_sequence_counts(sl, max_timestep):

    sc = [tf.reduce_sum(tf.cast(tf.greater_equal(sl, 1), tf.int32))]
    
    for i in range(1,max_timestep) :
        sc = tf.concat([sc, [tf.reduce_sum(tf.cast(tf.greater_equal(sl, i+1), tf.int32))]], axis=-1)
    
    return sc

def metric_timestep_accuracy(acc_tensor, eval_metric_ops) :
    
    for i in range(acc_tensor.shape[0]) :
        eval_metric_ops["acc"+str(i)]= tf.metrics.mean(tf.gather(acc_tensor,i))
    
    return eval_metric_ops

def metric_timestep_acc_by_top_qty(metric_tensor, eval_metric_ops) :
    
    for i in range(metric_tensor.shape[0]) :
        eval_metric_ops["acc_by_tq"+str(i)]= tf.metrics.mean(tf.gather(metric_tensor,i))
    
    return eval_metric_ops
    
def metric_timestep_acc_by_times_bought(metric_tensor, eval_metric_ops) :
    
    for i in range(metric_tensor.shape[0]) :
        eval_metric_ops["acc_by_tb"+str(i)]= tf.metrics.mean(tf.gather(metric_tensor,i))
    
    return eval_metric_ops

def metric_sequence_counts(metric_tensor, eval_metric_ops) :
    
    for i in range(metric_tensor.shape[0]) :
        eval_metric_ops["sc"+str(i)]= tf.metrics.mean(tf.gather(metric_tensor,i))
    
    return eval_metric_ops

def generate_output(top_k, pred_sequences, index):
    
    begin= tf.constant([0,0],dtype=tf.int32)
    end = tf.concat([[tf.constant(1, dtype=tf.int32)],index],0)
    valid_pred_seqs = tf.slice(pred_sequences,begin,end)
    valid_pred_seqs.set_shape([1,None])
    
    export_predictions = {
        "prod_indices": top_k,
        "top_ts_preds": valid_pred_seqs
   #     "top_ts_preds":pred_sequences
    }
    
    export_outputs = {
        "inference_data": PredictOutput(export_predictions)
    }
    
    return export_outputs

def model_fn(features, labels, mode, hyperparameters):
    
    with tf.name_scope("RNN_layers"):
        
        if (mode == tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL):        
            hyperparameters["DROPOUT_RNN_STATE_KEEP_PROB"]=1
            hyperparameters["DROPOUT_RNN_INPUT_KEEP_PROB"]=1
            hyperparameters["L2_REG_DENSE"] = 0
    
        cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hyperparameters["LSTM_UNITS"], 
                                                                          reuse=tf.get_variable_scope().reuse),
                                            state_keep_prob=hyperparameters["DROPOUT_RNN_STATE_KEEP_PROB"],
                                            input_keep_prob=hyperparameters["DROPOUT_RNN_INPUT_KEEP_PROB"])
   
    with tf.name_scope('inputs'):
        # [mini-batch, time_step, feature dims]
        index = features[SL_TENSOR_NAME]
        x = features[INPUT_TENSOR_NAME]
        y = labels
        sc = _calculate_sequence_counts(index,hyperparameters["MAX_TIMESTEP"])
                 
    with tf.name_scope("RNN_forward"):
        out, _ = tf.nn.dynamic_rnn(cell, x, sequence_length=index, dtype=tf.float32)  
            
        dense_decoder= Dense(hyperparameters["OUTPUT_SIZE"], \
                            kernel_initializer='glorot_uniform', bias_initializer='zeros',
                            kernel_regularizer= l2(hyperparameters["L2_REG_DENSE"]))
            
        fc_o1 = TimeDistributed(dense_decoder, \
                                input_shape = (None, hyperparameters["MAX_TIMESTEP"], hyperparameters["LSTM_UNITS"])) (out)
            
    with tf.name_scope('predictions'):         
        
        a_o1 = tf.nn.softmax(fc_o1)
        predictions = tf.argmax(a_o1, axis=2, name="inference_predictions")
        one_hot_predictions = tf.one_hot(predictions, hyperparameters["OUTPUT_SIZE"]) 
        tf.summary.histogram('predictions', predictions)
        
        if mode == tf.estimator.ModeKeys.PREDICT:
     
            last_time_slice = tf.subtract(index,1)
            pred_next_time_slice = tf.gather(a_o1, last_time_slice, axis=1)
            k_v, k_i = tf.nn.top_k(pred_next_time_slice, k=hyperparameters["K_PREDICTIONS"], sorted=True)
            
            export_outputs= generate_output(k_i, predictions, index)
                            
            return tf.estimator.EstimatorSpec(mode=mode, 
                                              predictions={"predictions": k_i},
                                              export_outputs= export_outputs)
            
    with tf.name_scope('loss'):
        weighted_targets = tf.multiply(y, hyperparameters["MSE_POS_WEIGHT"])
        
        mse = tf.reduce_sum(tf.losses.mean_squared_error(
                                predictions = tf.reshape(fc_o1, [-1, hyperparameters["OUTPUT_SIZE"]]),
                                labels = tf.reshape(weighted_targets, [-1, hyperparameters["OUTPUT_SIZE"]])))

        loss= tf.divide(mse,tf.cast(hyperparameters["BATCH_SIZE"], tf.float32), name="metric_loss")
        tf.summary.scalar('loss', loss)
        
    with tf.name_scope('train') :
        optimizer= tf.train.AdamOptimizer(learning_rate=hyperparameters["LEARNING_RATE"])
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())
            
    with tf.name_scope('validation') :
        correct_predictions = tf.reduce_sum(tf.cast(  \
        tf.greater(tf.multiply(one_hot_predictions, y),0.), tf.float32), axis=2, name="validate_correct")
        correct_pred_indices = tf.multiply(tf.cast(predictions, tf.float32),  \
                                                        correct_predictions, name="validate_correct_indices")
                        
        total_correct = tf.reduce_sum(correct_predictions, axis=0)  
        valid_predictions_per_slice =  tf.cast(tf.maximum(sc, tf.ones(sc.shape, tf.int32)), tf.float32)
        accuracy = tf.divide(total_correct, valid_predictions_per_slice, name="metric_acc")
        tf.summary.histogram('accuracy', accuracy)
       
        eval_metric_ops = {}
        eval_metric_ops = metric_timestep_accuracy(accuracy,eval_metric_ops)
        if mode == tf.estimator.ModeKeys.TRAIN:
            
            training_hooks = [tf.train.LoggingTensorHook(tensors = {"accuracy": accuracy},every_n_iter=100)]
               
            return tf.estimator.EstimatorSpec(mode=mode, 
                                              loss=loss, 
                                              train_op=train_op,
                                              training_hooks=training_hooks)
                    
        predict_by_top_qty = tf.reduce_sum(tf.reduce_sum(tf.cast(  \
                tf.greater(tf.multiply(tf.one_hot(hyperparameters["TOP_PROD_BY_QTYBOUGHT"],
                                                  hyperparameters["OUTPUT_SIZE"]), y),0.), \
                                                     tf.float32), axis=2, name="validate_correct_by_top_qty"), axis=0)
            
        predict_by_times_bought = tf.reduce_sum(tf.reduce_sum(tf.cast(  \
                tf.greater(tf.multiply(tf.one_hot(hyperparameters["TOP_PROD_BY_TIMESBOUGHT"], 
                                                hyperparameters["OUTPUT_SIZE"]),y),0.), \
                                                          tf.float32), axis=2, name="validate_correct_by_times_bought"), axis=0)

        acc_by_top_qty = tf.divide(predict_by_top_qty,valid_predictions_per_slice, name="metric_acc_by_top_qty")
        acc_by_times_bought = tf.divide(predict_by_times_bought,valid_predictions_per_slice, name="metric_acc_by_times_bought")
           
        if mode == tf.estimator.ModeKeys.EVAL:
             
            eval_metric_ops = metric_timestep_acc_by_top_qty(acc_by_top_qty,eval_metric_ops)
            eval_metric_ops = metric_timestep_acc_by_times_bought(acc_by_times_bought,eval_metric_ops)
            eval_metric_ops = metric_sequence_counts(sc,eval_metric_ops)
            
            eval_hooks = [tf.train.LoggingTensorHook(tensors = {"accuracy": accuracy, "sc": sc},every_n_iter=100)]
                   
            return tf.estimator.EstimatorSpec(mode=mode, 
                                              loss=loss,
                                              evaluation_hooks= eval_hooks,
                                              eval_metric_ops=eval_metric_ops)
          
    with tf.name_scope('tensorboard') :
        merged = tf.summary.merge_all()