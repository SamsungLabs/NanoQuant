# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch

from ..core.compress_model import compress_block_recon, compress_model_recon
from ..core.importance import collect_stats, get_shrunk_stats, register_stats
from ..utils.data_utils import get_calib_loader, prepare_dataset
from ..utils.load_utils import (get_compressed_state_dict, load_compressed_model, load_model, load_tokenizer)


class AutoNQModel():
    def __init__(self):
        self.model = None
        self.quant_config = None

    @classmethod
    def from_pretrained(cls, model_id: str, qmodel_path: str, dtype: torch.dtype = torch.bfloat16,
                        device_map: str = "cuda", quant_config: dict = {}):
        """
        Load quantized checkpoint if exists,
        otherwise quantize the model.
        """
        instance = cls()

        # check if qmodel_path exists
        if qmodel_path:
            if os.path.isfile(qmodel_path):
                model = instance.load_model(model_id, qmodel_path, quant_config, device_map, dtype)
                return model

        # quantize model
        model = instance.quantize_model(model_id, quant_config)
        # save model
        if qmodel_path:
            instance.save_model(model, qmodel_path)
        # return quantized model
        return model

    def quantize_model(self, model_id, quant_config):
        """
        Quantize model
        """
        # load model and fp_model
        device_map = quant_config.get('device_map', 'cpu')
        model = load_model(model_id, quant_config['seqlen'], device_map=device_map)
        fp_model = load_model(model_id, quant_config['seqlen'], device_map=device_map)

        # load dataloader
        data = prepare_dataset(model_id, quant_config)
        tokenizer = load_tokenizer(model_id)
        dataloader = get_calib_loader(data, tokenizer, quant_config['num_calib_samples'], quant_config['seed'],
                                      quant_config['seqlen'])

        # get importance via calibration
        raw_stats = collect_stats(model, dataloader, "cuda", strategy=quant_config['calib_strategy'])
        shrunk_stats = get_shrunk_stats(raw_stats, shrinkage=quant_config['calib_shrinkage'])
        model = register_stats(model, shrunk_stats)

        model = compress_block_recon(model, fp_model, dataloader, quant_config)
        model = compress_model_recon(model, fp_model, dataloader, quant_config)

        return model

    def load_model(self, model_id, qmodel_path, quant_config, device_map, dtype):
        """
        Load quantized model.
        """
        return load_compressed_model(model_name_or_path=model_id, checkpoint_path=qmodel_path,
                                     seqlen=quant_config['seqlen'], has_mid_scale=(quant_config['admm_type'] == 'dbf'),
                                     device=device_map, dtype=dtype)

    def save_model(self, model, qmodel_path):
        """
        Save quantized model.
        """
        output_path = Path(qmodel_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        compressed_state_dict = get_compressed_state_dict(model)
        torch.save(compressed_state_dict, qmodel_path)
