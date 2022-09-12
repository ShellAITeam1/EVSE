import logging
import sys

logger = logging.getLogger(__name__)

msg_format = "%(asctime)s %(levelname)-8s %(filename)s:%(lineno)s - %(funcName)4s: %(message)s"
logging.basicConfig(stream=sys.stdout, format=msg_format, datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
