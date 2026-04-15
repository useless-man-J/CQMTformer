import torch.nn as nn
class Patch(nn.Module):
    def __init__(self,seq_len,patch_len,stride,padding_patch):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((seq_len - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
    def forward(self,x):
        if self.padding_patch == 'end':
            x= self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [bs, nvars, patch_num, patch_len]
        x = x.permute(0, 1, 3, 2)  # [bs, nvars, patch_len, patch_num]
        return x