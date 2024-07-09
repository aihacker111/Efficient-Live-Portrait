# coding: utf-8

import tyro
from EfficientLivePortrait.config.argument_config import ArgumentConfig
from EfficientLivePortrait.config.inference_config import InferenceConfig
from EfficientLivePortrait.live_portrait_pipeline import LivePortraitPipeline


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig

    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg
    )

    # run
    live_portrait_pipeline.execute(args)


if __name__ == '__main__':
    main()
