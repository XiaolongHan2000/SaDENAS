import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#from architecture       import Architecture
from copy                     import deepcopy
from cell_operationsNAS201    import ResNetBasicblock
from genotypesNAS201          import Structure
from search_cellsNAS201          import NAS201SearchCell as SearchCell

class TinyNetwork(nn.Module):
    
  def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats):
    super(TinyNetwork, self).__init__()
    self._C        = C
    self._layerN   = N
    self.max_nodes = max_nodes
    self.search_space = search_space
    self.stem = nn.Sequential(nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C))
  
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev, num_edge, edge2index = C, None, None
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2)  # 不去搜的cell
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
        if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self.op_names   = deepcopy( search_space )
    self._Layer     = len(self.cells)
    self.edge2index = edge2index
    self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    # Initializing the architecture parameter
    self.num_edge = num_edge
    self.search_space = search_space
    self.arch_parameters = nn.Parameter(torch.rand(self.num_edge, len(self.search_space)), requires_grad = False)
    #self.arch_parameters = nn.Parameter( 1e-3*torch.randn(num_edge, len(search_space)), requires_grad = False)
    #self.arch_parameters = Architecture(num_edges = num_edge, search_space = search_space)

  def get_weights(self):
    '''To get the values of the weigths/parameters of the one shot model'''
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
    xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

  def random_alphas(self):
    '''Generate random values for the architecture parameters'''
    try:
      tmp = torch.rand(self.num_edge, len(self.search_space))
      self.update_alphas(tmp)
    except:
      tmp = torch.rand(self.num_edge, len(self.search_space)).cuda()
      self.update_alphas(tmp)

  def get_alphas(self):
    '''Returns the architecture weights'''
    return [self.arch_parameters]

  def update_alphas(self, alphas):
    '''Update the architecture weights'''
    #assert isinstance(alphas, Architecture) , 'Input alphas is wrong type'
    assert isinstance(alphas, torch.Tensor) , 'Input alphas has to be tensor type'
    assert alphas.device == self.get_alphas()[0].device, "Given tensor has to be in the same device"
    assert alphas.size() == self.get_alphas()[0].size(), "Given tensor size must be {}".format(self.get_alphas()[0].size())
    self.get_alphas()[0].data.copy_(alphas)

  def discretize(self):
    ''' Discretize the alphas of the model'''
    alphas_softmax  = nn.functional.softmax(self.get_alphas()[0], dim=-1)
    index = alphas_softmax.max(-1, keepdim=True)[1]
    one_h = torch.zeros_like(alphas_softmax).scatter_(-1, index, 1.0)
    self.update_alphas(one_h)

    return one_h
  
  def check_alphas(self, alphas):
    '''Verify if the architecture parameter is equal to the given alphas'''
    assert isinstance(alphas, torch.Tensor) , 'Input alphas has to be tensor type'
    #return np.all(self.get_alphas()[0] == alphas.get())
    return torch.all(self.get_alphas()[0] == alphas).item()

  def show_alphas(self):
    '''Returns the architecture parameter in string format'''
    with torch.no_grad():
      #return 'arch-parameters :\n{:}'.format( nn.functional.softmax(self.arch_parameters.get_tensor(), dim=-1).cpu() )
      return 'arch-parameters :\n{:}'.format( self.get_alphas()[0].cpu() )

  def show_alphas_dataframe(self, show_sum = False):
    '''Returns the architecture weights in two pandas Dataframe.
      First dataframe is just the architecture weight and the
      second dataframe is the one hot representation of the max values in each row in first dataframe
    '''
    df = pd.DataFrame(columns = self.search_space, index = list(self.edge2index.keys()))
    df_softmax = pd.DataFrame(columns = self.search_space, index = list(self.edge2index.keys()))
    df_max = pd.DataFrame(columns = self.search_space, index = list(self.edge2index.keys()),
                  data = np.zeros(self.get_alphas()[0].shape, dtype = np.int64))

    alphas = self.get_alphas()[0].detach().cpu().numpy()
    for index, key in enumerate(self.edge2index.keys()):
      #df.loc[key] = self.arch_parameters.get()[index]
      df.loc[key] = alphas[index]
    
    alphas_softmax  = nn.functional.softmax(self.arch_parameters, dim=-1).detach().cpu().numpy()
    #alphas_softmax = self.arch_parameters.apply_softmax(self.get_alphas()[0])
    for index, key in enumerate(self.edge2index.keys()):
      df_softmax.loc[key] = alphas_softmax[index]
    
    max_idx = df.astype(float).idxmax(axis = 1)
    for column, row in enumerate(max_idx.index):
      # row = int(row)
      # column = int(column)
      # df_max.iloc[row, max_idx[column]] = 1
      df_max.loc[row][max_idx[column]] = 1
    
    if show_sum:
      df_softmax['sum'] = df_softmax.sum(axis = 1)

    return df, df_max, df_softmax


  def get_message(self):
    '''Returns the string format of details of one shot model'''
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def genotype(self):
    '''Returns the architecture from the one shot model architecture parameter'''
    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        with torch.no_grad():
          weights = self.arch_parameters[ self.edge2index[node_str] ]
          op_name = self.op_names[ weights.argmax().item() ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return Structure( genotypes )

  def forward(self, inputs):
    alphas  = nn.functional.softmax(self.get_alphas()[0], dim=-1)
    #print(alphas)
    #alphas  = self.arch_parameters
    #alphas = torch.tensor(self.get_alphas()[0])

    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        feature = cell(feature, alphas)
      else:
        feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits
