import sys

import logging
logger = logging.getLogger(__name__)

def logging_setup():
    """ Set logging
    """
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    _set_log("cogsciabc", logging.INFO, [ch])
    _set_log("elfi", logging.INFO, [ch])
    _set_log("elfi.methods", logging.DEBUG, [ch])
    _set_log("elfie", logging.INFO, [ch])
    _set_log("elfie.mpi", logging.INFO, [ch])
    _set_log("elfie.acquisition", logging.DEBUG, [ch])
    _set_log("elfie.bolfi_extensions", logging.DEBUG, [ch])
    _set_log("elfirl", logging.INFO, [ch])

def _set_log(name, level, handlers):
    l = logging.getLogger(name)
    l.setLevel(level)
    l.propagate = False
    l.handlers = handlers

