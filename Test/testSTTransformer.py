from functools import reduce
import os
import sys

print(sys.path)
sys.path.append(os.path.join(sys.path[0], ".."))
print(sys.path)
from STTransformer import InversePatchify, STTransformer, TemporalAttention, TemporalMLP
import unittest
import torch


class TestModule(unittest.TestCase):
    def test_TemporalMLP(self):
        temporalMlp = TemporalMLP(10, 10, 20)
        x = torch.randn((10, 10))
        out = temporalMlp(x)
        self.assertEquals(out.shape, torch.randn((10, 20)).shape)

    def test_InversePatchfiy(self):
        map_size = [32, 32]
        patch_size = [4, 4]
        token_dim = 256
        num_channels = 3
        inversePatchify = InversePatchify(map_size, patch_size, token_dim, num_channels)
        num_tokens = reduce(
            lambda x, y: x * y, [ms // ps for ms, ps in zip(map_size, patch_size)]
        )
        x = torch.randn((1, num_tokens, token_dim))
        out = inversePatchify(x)
        self.assertEqual(out.shape, torch.randn((1, num_channels, *map_size)).shape)

    def testTemporalAttention(self):
        map_size = [32, 32]
        time_embedding_dim = 48
        timeAtten = TemporalAttention(time_embedding_dim, map_size)
        x = torch.randn((1, time_embedding_dim))
        out = timeAtten(x)
        self.assertEqual(out.shape, torch.randn((1, 1, *map_size)).shape)

    def testSTTransformer(self):
        batch_size, len_c, len_t, height, width = 88, 2, 2, 32, 32
        patch_size = [4, 8]

        sttransformer = STTransformer(
            map_height=height,
            map_width=width,
            patch_size=patch_size,
            len_closeness=len_c,
            len_trend=len_t,
            close_dim=1024,
            trend_dim=1024,
            close_depth=4,
            trend_depth=4,
            close_head=2,
            trend_head=2,
            close_mlp_dim=2048,
            trend_mlp_dim=2048,
            nb_flow=2,
            pre_conv=True,
            shortcut=True,
            conv_channels=64,
            drop_prob=0.1,
            time_class=7 + 48,  # num_of_day_of_week(7) + num_of_hour_of_day(24/48)
            temporal_hidden_dim=2048,
            post_num_channels=10,
        )
        xc = torch.randn((batch_size, 2, len_c, height, width))
        xt = torch.randn((batch_size, 2, len_t, height, width))
        flow_prediction, day_of_week_label, time_of_day_label = sttransformer(xc, xt)
        print(flow_prediction.shape)
        print(day_of_week_label.shape)
        print(time_of_day_label.shape)


if __name__ == "__main__":
    unittest.main()
