import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from layers import EmbeddingLayer, Encoder

class Agent(nn.Module):