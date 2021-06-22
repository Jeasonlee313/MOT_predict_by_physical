import torch
from model import Net


def cvt_model():
    print("===> Loading model")
    model = Net(reid=True)
    state_dict = torch.load('./checkpoint/ckpt.t7', map_location=lambda storage, loc: storage)['net_dict']
    model.load_state_dict(state_dict)  # 从字典中依次读取，具体值查看字典更改
    print('===> Load last checkpoint data')

    # 模型转换，Torch Script
    model.eval()
    example = torch.rand(4,3,128,64)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("deep_model.pt")
    print("Export of model.pt complete!")

if __name__ == '__main__':
    cvt_model()