# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""

import torch
import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf

# Configure TensorFlow and PyTorch for CUDA compatibility
def configure_gpu_memory():
  """Configure GPU memory to prevent CUDA conflicts."""
  # TensorFlow GPU configuration
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Enable memory growth for TensorFlow
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      # Set virtual GPU memory limit if needed
      tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]  # 8GB limit
      )
      logging.info(f"TensorFlow configured with {len(gpus)} GPU(s)")
    except RuntimeError as e:
      logging.warning(f"GPU configuration error: {e}")

  # PyTorch CUDA configuration
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Set PyTorch memory fraction
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    logging.info(f"PyTorch configured with CUDA {torch.version.cuda}")
    logging.info(f"Available GPUs: {torch.cuda.device_count()}")

  # Set environment variables for CUDA
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
  os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU only

# Configure GPU memory before any model loading
configure_gpu_memory()

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("assetdir", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) +
                    "/assets/stats/", "The folder name for storing evaluation results")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
  tf.io.gfile.makedirs(FLAGS.workdir)
  with open(os.path.join(FLAGS.workdir, 'config.txt'), 'w') as f:
    # f.write(str(FLAGS.config.to_dict()))
    for k, v in FLAGS.config.to_dict().items():
      f.write(str(k) + '\n')
      print(type(v))
      if type(v) == dict:
        for k2, v2 in v.items():
          f.write('> ' + str(k2) + ': ' + str(v2) + '\n')
      f.write('\n\n')
  if FLAGS.mode == "train":
    # Create the working directory
    tf.io.gfile.makedirs(FLAGS.workdir)
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    if os.path.exists(os.path.join(FLAGS.workdir, 'stdout.txt')):
      gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'a')
    else:
      gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the training pipeline
    run_lib.train(FLAGS.config, FLAGS.workdir, FLAGS.assetdir)
  elif FLAGS.mode == "eval":
    eval_dir = os.path.join(FLAGS.workdir, FLAGS.eval_folder)
    tf.io.gfile.makedirs(eval_dir)
    stdout_name = 'evaluation_history'
    if os.path.exists(os.path.join(FLAGS.workdir, f'{stdout_name}.txt')):
      gfile_stream = open(os.path.join(FLAGS.workdir, f'{stdout_name}.txt'), 'a')
    else:
      gfile_stream = open(os.path.join(FLAGS.workdir, f'{stdout_name}.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the evaluation pipeline
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.assetdir, FLAGS.eval_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
