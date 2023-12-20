Utils
======

This module contains a collection of utility classes and functions that are used throughout the library.

.. important::
    The most important module is `main.py`, which is the entry point for the library. 
    It contains the `main` function, which is called when the library is run as a script. 
    This function is responsible for parsing the command line arguments and calling the appropriate 
    functions to perform train and validation.

Other notable modules are:  

- `args.py`: contains all the **global** arguments. For **model-specific** arguments, see the `parse_args` function in the corresponding model file (under `models/<MODEL NAME>`).  

- `buffer.py`: contains the `Buffer` class, which is used to store the data for the replay buffer.  

- `training.py`: contains the `train` function, which is responsible for training the model, and the `evaluate` function, which is responsible for evaluating the model. The `train` function iterates over all the tasks and supports `3` utility functions: `begin_task`, `end_task`, and `observe`:

  - `begin_task`: called at the beginning of each task. It is useful if the model needs to set its internal state before     starting the task (e.g., calculating some preliminary statistics or adding new parameters for the new task).  

  - `end_task`: called at the end of each task. This function can be used to save the model after each task or perform some last-minute operations before the task ends (for example, in the case of `gdumb` it can be used to train on the data currently stored in the buffer).  

  - `observe`: called at each training step. It should contain *all the logic to train the model on the current batch*, including updating the replay buffer and the target network (if applicable). It should also return the loss value for the current batch.  

- `conf.py`: contains some utility functions such as the default path where to download the datasets (`base_path`) and the default device to use (`get_device`). 