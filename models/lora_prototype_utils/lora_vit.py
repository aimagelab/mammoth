from backbone.vit import VisionTransformer as MammothVP


class VisionTransformer(MammothVP):
    """
    Vision Transformer with support for LoRA
    """

    def __init__(self, *args, **kwargs):
        # block_fn = partial(MammothViTBlock, mlp=Mlp, attn=Attention)
        super().__init__(*args, **kwargs, use_lora=True)  # block_fn=block_fn)

    @classmethod
    def from_mammoth(cls, backbone: MammothVP):
        """
        Load weights from a Mammoth model
        """
        vit_model = cls(
            embed_dim=backbone.embed_dim,
            depth=backbone.depth,
            num_heads=backbone.num_heads,
            drop_path_rate=backbone.drop_path_rate,
            num_classes=backbone.num_classes
        )
        load_dict = backbone.state_dict()
        for k in list(load_dict.keys()):
            if 'head' in k:
                del load_dict[k]
        missing, unexpected = vit_model.load_state_dict(load_dict, strict=False)
        assert len([m for m in missing if 'head' not in m]) == 0, f"Missing keys: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
        return vit_model
