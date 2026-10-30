"""Image preview, edge and mask workflows described as operations and a DAG."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..sarp import Graph, Node, Operation


def grayscale(image):
    return image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114


def gaussian(image, passes=2):
    value = image
    weights = (1., 4., 6., 4., 1.)
    for _ in range(passes):
        for axis in (0, 1):
            padding = [(0, 0), (0, 0)]
            padding[axis] = (2, 2)
            padded = np.pad(value, padding, mode="edge")
            parts = []
            for i, weight in enumerate(weights):
                section = [slice(None), slice(None)]
                section[axis] = slice(i, i + value.shape[axis])
                parts.append(padded[tuple(section)] * (weight / 16.))
            value = sum(parts)
    return value


def edges(image):
    p = np.pad(image, 1, mode="edge")
    gx = (-p[:-2, :-2] + p[:-2, 2:] - 2 * p[1:-1, :-2]
          + 2 * p[1:-1, 2:] - p[2:, :-2] + p[2:, 2:])
    gy = (-p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]
          + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
    return np.sqrt(gx * gx + gy * gy)


@dataclass(frozen=True)
class ImageRequest:
    images: Mapping[str, np.ndarray]
    branches: tuple[str, ...] = ("edges", "mask", "preview")
    passes: int = 2


class ImagePipelineAdapter:
    operations = {
        "grayscale": Operation(grayscale, pure=True),
        "gaussian": Operation(gaussian, pure=True),
        "edges": Operation(edges, pure=True),
        "mask": Operation(lambda x, level: (x >= level).astype(np.uint8), pure=True),
        "preview": Operation(lambda x, gain: np.clip(x * gain, 0., 1.), pure=True),
    }

    def build(self, request: ImageRequest) -> Graph:
        if not request.branches or not set(request.branches) <= {"edges", "mask", "preview"}:
            raise ValueError("unsupported image output")
        if len(request.branches) != len(set(request.branches)) or request.passes < 1:
            raise ValueError("invalid image pipeline configuration")
        nodes, outputs = [], []
        for name, image in request.images.items():
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("expected H x W x 3 image")
            pixels = int(np.prod(image.shape[:2]))
            for branch in request.branches:
                gray, blur, output = f"{name}/{branch}/gray", f"{name}/{branch}/blur", f"{name}/{branch}"
                nodes.append(Node(gray, "grayscale", (name,), work=5 * pixels))
                nodes.append(Node(blur, "gaussian", (gray,), {"passes": request.passes},
                                  20 * request.passes * pixels))
                params = {"level": 0.5} if branch == "mask" else ({"gain": 1.2} if branch == "preview" else {})
                nodes.append(Node(output, branch, (blur,), params,
                                  (24 if branch == "edges" else 2) * pixels))
                outputs.append(output)
        return Graph("image-pipeline-v1", request.images, tuple(nodes), tuple(outputs))


def construct(seed=7, *, images=6, size=192, duplicate=True) -> ImageRequest:
    rng = np.random.default_rng(seed)
    y, x = np.indices((size, size))
    inputs = {}
    for i in range(images):
        texture = np.stack([np.sin(x / (7 + c) + y / (11 + i)) for c in range(3)], axis=-1)
        inputs[f"image_{i}"] = np.clip(0.5 + 0.25 * texture + rng.normal(0, .08, texture.shape), 0., 1.)
    return ImageRequest(inputs, ("edges", "mask", "preview") if duplicate else ("edges",))


def changed(request: ImageRequest, fraction: float) -> ImageRequest:
    images = dict(request.images)
    for name in list(images)[:int(np.ceil(len(images) * fraction))]:
        images[name] = np.clip(images[name] + 0.0125, 0., 1.)
    return ImageRequest(images, request.branches, request.passes)


def expected(request: ImageRequest) -> dict[str, np.ndarray]:
    outputs = {}
    for name, image in request.images.items():
        blur = gaussian(grayscale(image), request.passes)
        for branch in request.branches:
            outputs[f"{name}/{branch}"] = (edges(blur) if branch == "edges" else
                (blur >= .5).astype(np.uint8) if branch == "mask" else np.clip(blur * 1.2, 0., 1.))
    return outputs
