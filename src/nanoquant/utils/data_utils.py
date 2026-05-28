# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import random

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from tqdm import trange
from transformers.tokenization_utils_base import BatchEncoding

from ..utils.load_utils import load_tokenizer
from ..utils.utils import set_seed


def get_calib_loader(dataset_path, tokenizer, n_samples=128, seed=0, seqlen=2048):
    """
    Creates a dataloader for calibration.
    """
    if type(dataset_path) == "str":
        print(f"Loading dataset from disk: {dataset_path}")
        ds = datasets.load_from_disk(dataset_path)
    else:
        ds = dataset_path

    set_seed(seed)
    inds = np.random.randint(0, len(ds), size=(n_samples, ))

    input_ids = [ds[int(i)]["input_ids"] for i in inds]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # Fallback if pad_token_id is still not set (though it should be in compress.py)
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        print(f"Warning: tokenizer.pad_token_id is None. Using {pad_token_id} for padding.")

    processed_ids = []
    for ids in input_ids:
        tensor_ids = torch.tensor(ids)
        if len(tensor_ids) > seqlen:
            processed_ids.append(tensor_ids[:seqlen])
        else:
            padding_needed = seqlen - len(tensor_ids)
            processed_ids.append(F.pad(tensor_ids, (0, padding_needed), value=pad_token_id))

    dataloader = torch.stack(processed_ids).long()
    print(f"Calibration dataloader created with shape: {dataloader.shape}")
    return dataloader


def get_test_loaders(name, seqlen=2048, model_name=''):
    """
    Loads standard evaluation datasets like Wikitext2 and C4.
    """
    tokenizer = load_tokenizer(model_name)

    if 'wikitext2' in name:
        # Correct dataset name to 'wikitext' and config to 'wikitext-2-raw-v1'
        testdata = datasets.load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    elif 'c4' in name:
        testdata = datasets.load_dataset('allenai/c4',
                                         data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                                         split='validation')
        val_texts = [testdata[i]['text'] for i in range(min(1100, len(testdata)))]
        testenc = tokenizer(' '.join(val_texts), return_tensors='pt')
        testenc = testenc.input_ids[:, :(256 * seqlen)]
    else:
        raise ValueError("Unsupported dataset")

    return None, testenc


def get_trainenc(dataset_path, nsamples=128, seed=0):
    """
    Loads the calibration dataset from a given path and prepares it for evaluation
    by concatenating samples. It returns a BatchEncoding object to be compatible
    with the evaluate_ppl function.
    """
    ds = datasets.load_from_disk(dataset_path)

    np.random.seed(seed)
    inds = np.random.choice(len(ds), nsamples, replace=False)

    input_ids_list = [ds[int(i)]["input_ids"] for i in inds]

    # Concatenate all lists of token IDs into one
    concatenated_ids = []
    for ids in input_ids_list:
        concatenated_ids.extend(ids)

    # Convert to a single PyTorch tensor
    input_ids_tensor = torch.tensor(concatenated_ids, dtype=torch.long).unsqueeze(0)

    # FIX: Wrap the tensor in a BatchEncoding object
    # This creates an object with an .input_ids attribute, which evaluate_ppl expects.
    trainenc = BatchEncoding({'input_ids': input_ids_tensor})

    return trainenc


def _generate_samples_from_dataset(dataset_name, dataset_config, num_samples_to_generate, tokenizer, seq_len,
                                   shuffle_buffer_size, seed):
    """
    Helper function to generate a specific number of samples from a single dataset.
    It includes special handling for 'wikitext' and generic handling for other streaming datasets.
    """
    from ..utils.utils import set_seed
    set_seed(seed)

    if num_samples_to_generate == 0:
        return []

    calibration_samples = []
    config_info = f"(config: {dataset_config})" if dataset_config else ""
    print(f"\nProcessing dataset: {dataset_name} {config_info}")
    print(f"Attempting to generate {num_samples_to_generate} samples...")

    # --- Special handling specifically for the 'wikitext' dataset ---
    # This logic is triggered only if the dataset name is an exact match.
    if dataset_name.lower() in ["wikitext", "salesforce/wikitext"]:
        print("Applying special handling for wikitext: concatenating all documents.")
        # Load the full dataset into memory (it's small)
        full_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        # Concatenate all text entries and tokenize once
        all_text = "\n\n".join(full_dataset["text"])
        trainenc = tokenizer(all_text, return_tensors="pt")

        # Check if the concatenated text is long enough
        if trainenc.input_ids.shape[1] <= seq_len:
            raise ValueError(f"Concatenated wikitext is not long enough for seq_len={seq_len}.")

        # Generate samples by randomly slicing the single large token tensor
        for _ in trange(num_samples_to_generate, desc="Slicing concatenated wikitext"):
            i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
            j = i + seq_len
            inp = trainenc.input_ids[:, i:j]
            assert inp.shape[1] == seq_len
            sample_dict = {'input_ids': inp.squeeze(0).tolist(), 'attention_mask': [1] * seq_len}
            calibration_samples.append(sample_dict)

        return calibration_samples
    elif "c4" in dataset_name.lower():
        traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                                 split='train')
        calibration_samples = []
        for _ in trange(num_samples_to_generate, desc="Slicing C4 documents"):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] > seq_len:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
            j = i + seq_len
            inp = trainenc.input_ids[:, i:j]
            assert inp.shape[1] == seq_len
            sample_dict = {'input_ids': inp.squeeze(0).tolist(), 'attention_mask': [1] * seq_len}
            calibration_samples.append(sample_dict)
        return calibration_samples

    # --- Generic handling for all other streaming datasets ---
    dataset = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True, trust_remote_code=True)

    # Shuffle the dataset using a buffer for improved randomness in document selection
    if shuffle_buffer_size > 0:
        print(f"Shuffling dataset with buffer size {shuffle_buffer_size}...")
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)

    # Filter for documents longer than the required sequence length
    def is_long_enough(sample):
        # Use 'text' column, common for these datasets
        return len(tokenizer(sample['text'], truncation=False)['input_ids']) > seq_len

    filtered_dataset = dataset.filter(is_long_enough)
    samples_iterator = iter(filtered_dataset.take(num_samples_to_generate * 5))  # Increased buffer

    pbar = trange(num_samples_to_generate, desc=f"Slicing documents from {dataset_name}")
    for _ in pbar:
        try:
            long_document = next(samples_iterator)
        except StopIteration:
            print(f"\nWarning: Could only find {len(calibration_samples)} long enough documents...")
            break

        # Tokenize the document and randomly slice a chunk of seq_len
        trainenc = tokenizer(long_document["text"], return_tensors="pt")

        if trainenc.input_ids.shape[1] <= seq_len:
            continue  # Skip if this specific document is somehow too short after all

        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len

        inp = trainenc.input_ids[:, i:j]
        assert inp.shape[1] == seq_len

        sample_dict = {'input_ids': inp.squeeze(0).tolist(), 'attention_mask': [1] * seq_len}
        calibration_samples.append(sample_dict)

    return calibration_samples


def prepare_dataset(model_id, quant_config):
    """
    Creates a calibration dataset from one or more datasets based on the given arguments
    and saves it as a Hugging Face Dataset directory.
    """
    set_seed(quant_config['seed'])

    if os.path.exists(quant_config['calib_dataset']):
        calib_data = datasets.load_from_disk(quant_config['calib_dataset'])
        return calib_data

    calib_data_type = quant_config['calib_dataset']
    assert calib_data_type in ['wikitext2', 'c4']

    tokenizer = load_tokenizer(model_id)

    # Use a list of Nones if no configs are provided
    dataset_name = "Salesforce/wikitext" if "wikitext2" == calib_data_type else "allenai/c4"
    dataset_config = "wikitext-2-raw-v1" if "wikitext2" == calib_data_type else "en"

    # Generate samples from each dataset
    samples = _generate_samples_from_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        num_samples_to_generate=quant_config['num_calib_samples'],
        tokenizer=tokenizer,
        seq_len=quant_config['seqlen'],
        shuffle_buffer_size=10000,
        seed=quant_config['seed'],
    )

    if not samples:
        raise ValueError(f"Could not generate any samples with sequence length {quant_config['seqlen']}.")

    # Shuffle the final combined list of samples
    print(f"\nShuffling the final combined dataset ({len(samples)} samples)...")
    random.shuffle(samples)

    # Create and save the final dataset to disk
    print(f"\nCreating Hugging Face Dataset...")
    hg_dataset = Dataset.from_list(samples)

    return hg_dataset
