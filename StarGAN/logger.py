import tensorflow as tf
from tensorboardX import SummaryWriter

class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir, exp_name):
        """Initialize summary writer."""
        #self.writer = tf.summary.FileWriter(log_dir)
        self.writer = SummaryWriter(log_dir, comment=exp_name)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        #self.writer.add_summary(summary, step)
        self.writer.add_scalar(tag, value, step)
