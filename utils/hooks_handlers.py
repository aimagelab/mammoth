import re
class HooksHandlerViTV2:
    def __init__(self, model) -> None:
       self.model = model
       self.block_outputs = []
       self.attentions = []
       self.hooks = []
 
       for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'VisionTransformer':
                self.hooks.append(module.register_forward_hook(self.process_outputs))
            # if len(re.findall(r'attn.proj$', name)) > 0:        # FIXARE: c'è scaled dot product attention nel vit di timm
            #     self.hooks.append(module.register_forward_hook(self.get_attention))
            if len(re.findall(r'blocks.\d+$', name)) > 0:       # FIXARE: prendere l'uscita del blocco
                self.hooks.append(module.register_forward_hook(self.get_block_outputs))
 
    def get_attention(self, module, input, output):
        self.attentions.append(input)
    
    def get_block_outputs(self, module, input, output):
        self.block_outputs.append(output)
    
    def reset(self):
        self.attentions = []
        self.block_outputs = []

    def process_outputs(self, module, input, output):
        res = {
            # 'attention_masks_heads': [x for x in self.attentions],
            'block_outputs': [x for x in self.block_outputs],
            'output': output
        }
        self.reset()
        return res

class HooksHandlerViT:
    def __init__(self, model) -> None:
       self.model = model
       self.attentions = []
 
       for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'VisionTransformer':
                module.register_forward_hook(self.add_intermediate_outputs)
            if 'attn_drop' in name:
                module.register_forward_hook(self.get_attention)
 
    def get_attention(self, module, input, output):
        self.attentions.append(output)
    
    def reset(self):
        self.attentions = []
    
    def add_intermediate_outputs_old(self, res):
        res['attention_masks'] = [x.mean(1) for x in self.attentions]
        self.reset()

    def add_intermediate_outputs(self, module, input, output):
        if isinstance(output, dict):
            output['attention_masks_heads'] = [x for x in self.attentions]
            output['attention_masks'] = [x.mean(1) for x in self.attentions]
        self.reset()