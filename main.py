import argparse
import json
import logging
import sys
import time
from functools import partial
from pathlib import Path

import tensorflow as tf

from inputs import *
from model_fns import *
from predict_fns import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

# This program was designed to function with multiple kinds of models, but currently only GPT2 is supported
# The first element in the tupel is the model function, the second is the function called when predicting
models = {
    "GPT2": (gpt2_model, gpt2_predict)
}

inputs = {
    "gpt2_huamei_corpus_seg_tsv_128k":gpt2_huamei_corpus_seg_tsv_128k,
    "openwebtext": openwebtext, # Standard OpenWebtext input
    "openwebtext_longbiased": openwebtext_longbiased, # OpenWebtext with a bias towards showing more long (>512 tokens) examples
    "openwebtext_long": openwebtext_long, # Openwebtext that only shows long examples
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu', type=str) # Name of TPU to train on, if any
    parser.add_argument('--model', type=str) # JSON file that contains model parameters
    parser.add_argument("--predict_file", type=str) # File to take as input for predict
    parser.add_argument("--predict_text", type=str) # Take string directly from args
    parser.add_argument("--top_k", type=int) # Top K truncation parameter for text generation
    args = parser.parse_args()

    # Get prediction text
    predict_mode = False
    if args.predict_file is not None:
        predict_mode = True
        with open(args.predict_file) as f:
            text = f.read()
    elif args.predict_text is not None:
        predict_mode = True
        text = args.predict_text
    elif args.predict_file is not None and args.predict_text is not None:
        print("ERROR: Specify exactly one of --predict_file and --predict_text!")
        sys.exit()


    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('logs/{}.log'.format(args.model)),
        logging.StreamHandler(sys.stdout)
    ]
    logger = logging.getLogger('tensorflow')
    logger.handlers = handlers

    # Read params of model
    with open(args.model, "r") as f:
        params = json.load(f)

    if not args.tpu is None:
        params["use_tpu"] = True
    else:
        params["use_tpu"] = False

    if args.top_k is not None:
        params["top_k"] = args.top_k

    if not "precision" in params.keys():
        params["precision"] = "float32" # Doesn't actually do anything since float32 is the default anyways. Only recognized other dtype is "bfloat16"

    if not "iterations" in params.keys():
        params["iterations"] = 1 # Because this controls how many samples are prefetched

    logger.info(params)

    model_fn = models[params["model"]][0]
    predict_fn = models[params["model"]][1]
    input_fn = inputs[params["input"]]

    if params["use_tpu"] and not predict_mode:
        # Resolve TPU cluster and runconfig
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(args.tpu)

        run_config = tf.contrib.tpu.RunConfig(
            model_dir=params["model_path"],
            cluster=tpu_cluster_resolver,
            save_checkpoints_secs=60*30,
            session_config=tf.ConfigProto(
                # allow_soft_placement=True,
                # log_device_placement=True
                ),
                tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=params["iterations"])
        )

        # Set up network
        network = tf.contrib.tpu.TPUEstimator(
                model_fn=model_fn,
                use_tpu=True,
                train_batch_size=params["train_batch_size"], # These are the global sizes, must be divisible by replicas
                eval_batch_size=params["eval_batch_size"],
                predict_batch_size=params["predict_batch_size"],
                config=run_config,
                params=params)

    else:
        # Non TPU setup
        if not predict_mode:
            params["batch_size"] = params["train_batch_size"]
        else:
            params["batch_size"] = params["predict_batch_size"]

            from models.gpt2 import encoder
            enc = encoder.get_encoder(params["encoder_path"])
            tokens = enc.encode(text)
            params["text_len"] = len(tokens)
            if params["text_len"] > 1024:
                params["text_len"] = 1024
                
        strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=2)
        
        run_config = tf.estimator.RunConfig(
            model_dir=params["model_path"],
            session_config=tf.ConfigProto(
                # log_device_placement=True,
                # allow_soft_placement=True
            )
            ,train_distribute=strategy
            ,save_checkpoints_secs=3600
            ,keep_checkpoint_max=40
        )

        network = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params=params)

    if predict_mode:
        logger.info("Generating predictions...")
        predict_fn(network, text, params)
        sys.exit()

    DEBUG_Inc_mem = True
    
    # Train eval loop
    while True:
        start = time.time()

        if(not DEBUG_Inc_mem):
            network.train(
                    input_fn=partial(input_fn, eval=False),
                    steps=params["train_steps"])


            end = time.time()
            logger.info("\nTrain loop took {:.2f}s\n".format(end-start))

            eval_result = network.evaluate(
               input_fn=partial(input_fn, eval=True),
               steps=params["eval_steps"])

            logger.info("\nEval Results: {}\n".format(str(eval_result)))
        else:
            
            """
            serving_feature_spec = tf.feature_column.make_parse_example_spec(
                  categorical_feature_a_emb)
            
            serving_input_receiver_fn = (
                  tf.estimator.export.build_parsing_serving_input_receiver_fn(
                  serving_feature_spec))

            exporter = tf.estimator.BestExporter(
                  name="best_exporter",
                  serving_input_receiver_fn=serving_input_receiver_fn,
                  exports_to_keep=10)
            """
            
            #用train_and_evaluate避免内存增长
            train_spec = tf.estimator.TrainSpec(input_fn=partial(input_fn, eval=False),max_steps=params["max_steps"])
            eval_spec = tf.estimator.EvalSpec(input_fn=partial(input_fn, eval=True),
                                              steps=params["eval_steps"],
                                                start_delay_secs=120,
                                                throttle_secs=3600,
                                                 #exporters=exporter
                                             )

            eval_result = tf.estimator.train_and_evaluate(network, train_spec, eval_spec)
            
            logger.info("\nEval Results: {}\n".format(str(eval_result)))
            end = time.time()
            logger.info("\nTrain and eval loop took {:.2f}s\n".format(end-start))
        
        global_step = network.get_variable_value("global_step")
        
        print("global_step:",global_step)
        if network.get_variable_value("global_step") > params["max_steps"]:
            logger.info("Done!")
            break
