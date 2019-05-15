import logging

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',level=logging.DEBUG)
x = 2
logging.debug(x)
logging.info(x)