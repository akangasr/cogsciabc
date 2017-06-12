
import logging
logger = logging.getLogger(__name__)

def logging_setup():
    """ Set logging
    """
    logger.setLevel(logging.INFO)
    model_logger = logging.getLogger("sdirl")
    model_logger.setLevel(logging.INFO)
    model_logger.propagate = False
    elfi_logger = logging.getLogger("elfi")
    elfi_logger.setLevel(logging.INFO)
    elfi_logger.propagate = False
    elfi_methods_logger = logging.getLogger("elfi.methods")
    elfi_methods_logger.setLevel(logging.DEBUG)
    elfi_methods_logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    model_logger.handlers = [ch]
    elfi_logger.handlers = [ch]
    elfi_methods_logger.handlers = [ch]
    logger.handlers = [ch]


