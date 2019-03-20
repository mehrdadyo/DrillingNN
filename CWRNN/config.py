
class Config(object):

    output_dir = "./output/"

    # Clockwork RNN parameters
    # periods       = [1, 2, 3, 4, 6, 8, 16, 32]  # old
    # periods       = [1, 2, 3, 4, 6, 12, 24, 28]   # 2nd best
    periods       = [1, 2, 3, 5, 8, 12, 18, 24]   # Official
    # periods       = [1, 2, 3, 5, 8, 12, 18, 24]   # test


    num_steps   = 0
    num_input   = 0
    num_hidden  = len(periods)*64    # official 64
    num_output  = 0

    # Optmization parameters
    num_epochs          = 100
    batch_size          = 128
    optimizer           = "adam"
    max_norm_gradient   = 10.0

    # Learning rate decay schedule
    learning_rate       = 1.00e-3
    learning_rate_decay = .997
    learning_rate_step  = 1000
    learning_rate_min   = 5.0e-4
