def set_settings(flags):
    flags.DEFINE_boolean('validate', True, 'validation process included or not')
    flags.DEFINE_string('train_image', '../data/train_image.npy', 'path to training dataset image')
    flags.DEFINE_string('train_label', '../data/train_label.npy', 'path to training dataset label')
    flags.DEFINE_string('valid_image', '../data/valid_image.npy', 'path to validation dataset image')
    flags.DEFINE_string('valid_label', '../data/valid_label.npy', 'path to validation dataset label')
    flags.DEFINE_string('checkpoint_dir_path', '../results/', 'path to checkpoint directory')
    flags.DEFINE_string('log_dir_path', '../results/logs/', 'path to log directory')
    flags.DEFINE_string('classes', '../data/polyp.names', 'path to classes file')
    flags.DEFINE_integer('epochs', 150, 'number of epochs')
    flags.DEFINE_integer('save_point', 150, 'save weights every')
    flags.DEFINE_integer('validation_point', 1, 'validate every')
    flags.DEFINE_integer('batch_size', 32, 'batch size')
    flags.DEFINE_integer('val_batch_size', 32, 'validation batch size')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
