#coding=utf-8
import time
import tensorflow as tf
import numpy as np
import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

if __name__ == '__main__':

    opt = TrainOptions().parse()
    tensorbord_log = Logger('../log_all/' + opt.name)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    if opt.which_step != 'latest':
        total_steps = int(opt.which_step)*opt.save_step_freq
    else:
        total_steps = 0
    count_print_loss = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1,):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += 1
            epoch_iter += opt.batchSize
            flg = model.set_input(data)
            if not flg:
                # 数据有奇数
                continue
            model.optimize_parameters(flag_shuffer=False)
            # tensorbord 可视化
            if total_steps % opt.display_freq == 0:
                for tag, value in model.get_current_losses().items():
                    tensorbord_log.scalar_summary(tag, value, total_steps+1)
                for tag, images in model.get_current_visuals().items():
                    images = images.cpu()[0].unsqueeze(0)
                    tensorbord_log.image_summary(tag,images, total_steps + 1)
                if total_steps % (opt.display_freq*2) == 0:
                    net_all = model.get_all_model()
                    for net_name in net_all:
                        for tag, value in net_all[net_name].named_parameters():
                            tag = net_name+'_'+tag
                            tag = tag.replace('.', '/')
                            tensorbord_log.histo_summary(tag, value.data.cpu().numpy(), total_steps + 1)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                print_current_losses(epoch, epoch_iter, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')
            if total_steps % opt.save_step_freq == 0:
                model.save_networks(total_steps//opt.save_step_freq)
            iter_data_time = time.time()
        model.save_networks('latest')

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        # model.update_learning_rate()
    model.save_networks('latest')