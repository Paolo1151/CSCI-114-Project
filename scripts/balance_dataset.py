import os
import sys
sys.path.append('..')

from modules.generation.generator import AccidentSeverityBalancer

bal = AccidentSeverityBalancer()
bal.generate_balanced_train()


