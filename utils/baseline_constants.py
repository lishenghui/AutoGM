"""Configuration file for common tf_models/experiments"""

MAIN_PARAMS = {
    'femnist': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (2000, 20, 20)
    },
    'shakespeare': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (20, 5, 2)
    },
}
"""dict: Specifies execution parameters (tot_global_rounds, eval_every_global_rounds, clients_per_round)"""

MODEL_PARAMS = {
    'femnist.cnn': (0.01, 62, 450),  # lr, num_classes, test_batch_size
    'cifar10.cnn': (0.01, 10, 64),  # lr, num_classes, test_batch_size
}
"""dict: Model specific parameter specification"""

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
LOSS_KEY = 'loss'
SET_KEY = 'set'
CORRUPTED_KEY = "corrupted"
