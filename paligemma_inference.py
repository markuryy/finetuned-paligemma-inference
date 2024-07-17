import os
import sys
import json
import random
import numpy as np
import glob
from PIL import Image
import jax
import jax.numpy as jnp
import tensorflow as tf
import sentencepiece
import ml_collections

# Add big_vision_repo to the Python path
if "big_vision_repo" not in sys.path:
    sys.path.append("big_vision_repo")

import big_vision.utils
import big_vision.sharding
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns

# Function to preprocess image
def preprocess_image(image, size=448):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    image = image[..., :3]
    assert image.shape[-1] == 3

    image = tf.constant(image)
    image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
    return image.numpy() / 127.5 - 1.0

# Function to preprocess tokens
def preprocess_tokens(tokenizer, prefix, suffix=None, seqlen=None):
    separator = "\n"
    tokens = tokenizer.encode(prefix, add_bos=True) + tokenizer.encode(separator)
    mask_ar = [0] * len(tokens)
    mask_loss = [0] * len(tokens)

    if suffix:
        suffix = tokenizer.encode(suffix, add_eos=True)
        tokens += suffix
        mask_ar += [1] * len(suffix)
        mask_loss += [1] * len(suffix)

    mask_input = [1] * len(tokens)
    if seqlen:
        padding = [0] * max(0, seqlen - len(tokens))
        tokens = tokens[:seqlen] + padding
        mask_ar = mask_ar[:seqlen] + padding
        mask_loss = mask_loss[:seqlen] + padding
        mask_input = mask_input[:seqlen] + padding

    return jax.tree_util.tree_map(np.array, (tokens, mask_ar, mask_loss, mask_input))

# Function to postprocess tokens
def postprocess_tokens(tokenizer, tokens):
    tokens = tokens.tolist()
    try:
        eos_pos = tokens.index(tokenizer.eos_id())
        tokens = tokens[:eos_pos]
    except ValueError:
        pass
    return tokenizer.decode(tokens)

# Function to perform inference
def make_predictions(params, tokenizer, model, data_iterator, data_sharding, batch_size=4, seqlen=512, sampler="greedy"):
    decode_fn = predict_fns.get_all(model)['decode']
    decode = functools.partial(decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id())
    outputs = []
    while True:
        examples = []
        try:
            for _ in range(batch_size):
                examples.append(next(data_iterator))
                examples[-1]["_mask"] = np.array(True)
        except StopIteration:
            if len(examples) == 0:
                return outputs

        while len(examples) % batch_size:
            examples.append(dict(examples[-1]))
            examples[-1]["_mask"] = np.array(False)

        batch = jax.tree_util.tree_map(lambda *x: np.stack(x), *examples)
        batch = big_vision.utils.reshard(batch, data_sharding)

        tokens = decode({"params": params}, batch=batch, max_decode_len=seqlen, sampler=sampler)

        tokens, mask = jax.device_get((tokens, batch["_mask"]))
        tokens = tokens[mask]
        responses = [postprocess_tokens(tokenizer, t) for t in tokens]

        for example, response in zip(examples, responses):
            outputs.append((example["image"], response))
            if len(outputs) >= num_examples:
                return outputs

# Function to load images from directory
def load_images_from_directory(image_dir, prefix="describe this image"):
    image_files = glob.glob(os.path.join(image_dir, '*.png')) + \
                  glob.glob(os.path.join(image_dir, '*.jpeg')) + \
                  glob.glob(os.path.join(image_dir, '*.jpg'))
    for image_file in image_files:
        image = Image.open(image_file)
        image = preprocess_image(image)
        tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, seqlen=SEQLEN)
        yield {
            "image": np.asarray(image),
            "text": np.asarray(tokens),
            "mask_ar": np.asarray(mask_ar),
            "mask_input": np.asarray(mask_input),
            "filename": image_file
        }

def main(model_path, image_dir, output_dir):
    # Load tokenizer
    tokenizer_path = "./paligemma_tokenizer.model"
    tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    # Load model
    model_config = ml_collections.FrozenConfigDict({
        "llm": {"vocab_size": 257_152},
        "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
    })
    model = paligemma.Model(**model_config)
    params = paligemma.load(None, model_path, model_config)

    # Shard parameters
    mesh = jax.sharding.Mesh(jax.devices(), ("data"))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    params_sharding = big_vision.sharding.infer_sharding(params, strategy=[('.*', 'fsdp(axis="data")')], mesh=mesh)
    params, treedef = jax.tree.flatten(params)
    sharding_leaves = jax.tree.leaves(params_sharding)
    trainable_mask = big_vision.utils.tree_map_with_names(lambda name, param: False, params)
    trainable_leaves = jax.tree.leaves(trainable_mask)
    for idx, (sharding, trainable) in enumerate(zip(sharding_leaves, trainable_leaves)):
        params[idx] = big_vision.utils.reshard(params[idx], sharding)
        params[idx] = params[idx].astype(jnp.float32) if trainable else params[idx]
        params[idx].block_until_ready()
    params = jax.tree.unflatten(treedef, params)

    # Make predictions
    data_iterator = load_images_from_directory(image_dir)
    predictions = make_predictions(params, tokenizer, model, data_iterator, data_sharding)

    # Save predictions
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for image, caption in predictions:
        image_filename = os.path.splitext(os.path.basename(image["filename"]))[0]
        caption_filename = os.path.join(output_dir, f"{image_filename}.txt")
        with open(caption_filename, "w") as f:
            f.write(caption)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PaliGemma Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the finetuned model")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images for inference")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated captions")

    args = parser.parse_args()

    main(args.model_path, args.image_dir, args.output_dir)
