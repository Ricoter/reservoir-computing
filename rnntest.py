import numpy as np
import torch



Wih = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=False)
Whh = torch.randn(H, H, device=device, dtype=dtype, requires_grad=False)
Who = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)


