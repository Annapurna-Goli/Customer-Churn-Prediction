import logging
import os


def setup_logging(script_name, log_dir="log_files"):
    """
    Sets up a logger that writes logs to a file and outputs to the console.

    :param script_name: The name used for the logger and the log file.
    :param log_dir: The subdirectory where logs will be stored (relative to the script).
    :return: A configured logging.Logger object.
    """
    # 1. Create directory if it doesn't exist (platform-independent)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. Configure the Logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding handlers multiple times if the function is called repeatedly
    if not logger.handlers:
        # 3. Create a common Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 4. Create File Handler (Appends to file by default)
        log_file_path = os.path.join(log_dir, f'{script_name}.log')
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 5. Create Stream Handler (Outputs to console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
