import sys
sys.path.append("/home/infres/belguith/PFE")
from model_run import config
from evaluate_model_run import test
test(config())
