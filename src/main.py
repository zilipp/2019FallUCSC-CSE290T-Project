import logging

# self defined module
from src.utils.utils import init_logger


# some config values
_user_logs_file = '..\\out\\logs\\user_logs\\logs.txt'  # User logging directory.


def main():
    init_logger(_user_logs_file)
    logging.info('Done!')


if __name__ == '__main__':
    main()
