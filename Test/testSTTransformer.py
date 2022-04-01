from functools import reduce
import os
import sys
print(sys.path)
sys.path.append(os.path.join(sys.path[0],".."))
print(sys.path)
from STTransformer import InversePatchify, TemporalAttention, TemporalMLP
import unittest
import torch

class TestModule(unittest.TestCase):
    def test_TemporalMLP(self):
        temporalMlp = TemporalMLP(10,10,20)
        x = torch.randn((10,10))
        out = temporalMlp(x)
        self.assertEquals(out.shape,torch.randn((10,20)).shape)
        
    def test_InversePatchfiy(self):
        map_size = [32,32]
        patch_size = [4,4]
        token_dim = 256
        num_channels = 3
        inversePatchify = InversePatchify(map_size,patch_size,token_dim,num_channels)
        num_tokens = reduce(lambda x,y: x*y, [ms // ps for ms,ps in zip(map_size,patch_size)])
        x = torch.randn((1,num_tokens,token_dim))
        out = inversePatchify(x)
        self.assertEqual(out.shape,torch.randn((1,num_channels,*map_size)).shape)
    def testTemporalAttention(self):
        map_size = [32,32]
        time_embedding_dim = 48
        timeAtten = TemporalAttention(time_embedding_dim,map_size)
        x = torch.randn((1,time_embedding_dim))
        out = timeAtten(x)
        self.assertEqual(out.shape,torch.randn((1,1,*map_size)).shape)
        
   
if __name__ == '__main__':
    unittest.main()