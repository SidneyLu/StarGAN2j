import os
import jittor as jt

"""This is the module for saving and loading checkpoints"""
"""检验点（权重）加载与读取模块"""

class CheckpointIO(object):
    def __init__(self, fname_template, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs

    #Regist network modules
    #向网络中注册模块以便加载参数
    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    #Save weights in .pth format for better compatibility
    #为兼容性考量，以.pth格式保存训练权重
    def save(self, step):
        fname = self.fname_template.format(step)
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            outdict[name] = module.state_dict()
        jt.save(outdict, fname)

    #Load pretrained weights
    #加载预训练权重，默认路径为expr/checkpoints
    def load(self, step):
        fname = self.fname_template.format(step)
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        state_dict = jt.load(fname)
        for name, module in self.module_dict.items():
            if hasattr(module, 'load_state_dict') and name in state_dict:
                module.load_state_dict(state_dict[name])