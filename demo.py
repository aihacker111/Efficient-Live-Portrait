import cv2
import tyro
from Face2Vid import ArgumentConfig, InferenceConfig, LivePortraitPipeline


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def main(cam=False):
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")

    args = tyro.cli(ArgumentConfig)

    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    live_portrait_pipeline = LivePortraitPipeline(inference_cfg=inference_cfg)
    if cam:
        cap = cv2.VideoCapture(0)
        s = '/Users/macbook/Downloads/Efficient-Face2Vid-Portrait/assets/examples/source/s0.jpg'
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            portrait, img_rgb = live_portrait_pipeline.render(source_image=s, source_motion=frame)
            portrait_rgb = cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR)
            cv2.imshow('img_rgb Image', img_rgb)
            cv2.imshow('Source Frame', frame)
            cv2.imshow('Live Portrait', portrait_rgb)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    else:
        d = '/Users/macbook/Downloads/Efficient-Face2Vid-Portrait/assets/examples/driving/d1.mp4'
        s = '/Users/macbook/Downloads/Efficient-Face2Vid-Portrait/assets/examples/source/s0.jpg'
        live_portrait_pipeline.render(source_image=s, source_motion=d, cam=cam)


if __name__ == '__main__':
    main(cam=False)