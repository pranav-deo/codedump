import torch
from typing import Iterable
from torchvision import transforms
from PIL import Image, ImageOps
import timeit

class PatchShuffle:
    def __init__(self, num_squares: Iterable, img_size: torch.Size) -> None:
        # Check if num_squares is a square of natural number
        assert len(num_squares) == 2, "num_squares should be (H,W)"
        assert img_size[0] % num_squares[0] == 0
        assert img_size[1] % num_squares[1] == 0
        self.num_squares = num_squares

    def __call__(
        self, image_tensor: torch.TensorType, bbox_coordinates: torch.TensorType
    ) -> torch.TensorType:
        """
        image_tensor: image tensor as CxHxW
        bbox_coordinates: Bounding box as 2x2 tensor, 2x[H,W] / 2x[y,x]
        """

        img_height, img_width = tuple(image_tensor.shape[1:])

        # get bbox points
        left_top = bbox_coordinates[0]
        right_bottom = bbox_coordinates[1]

        # Divide image into num_squares squares
        column_width = img_width // self.num_squares[1]
        row_height = img_height // self.num_squares[0]

        # get key boxes
        left_top_box = (left_top[1] // column_width, left_top[0] // row_height)
        right_bottom_box = (
            right_bottom[1] // column_width,
            right_bottom[0] // row_height,
        )

        # Get index of permutable boxes
        all_boxes = torch.arange(0, self.num_squares[0] * self.num_squares[1]).reshape(
            self.num_squares
        )
        all_boxes[
            int(left_top_box[1].item()) : int(right_bottom_box[1].item() + 1),
            int(left_top_box[0].item()) : int(right_bottom_box[0].item() + 1),
        ] = 0.0
        all_boxes = torch.flatten(all_boxes)
        permutable_boxes = all_boxes[torch.nonzero(all_boxes)]

        # Shuffle boxes
        shuffled_boxes = permutable_boxes[torch.randperm(permutable_boxes.shape[0])]

        # Make New image and copy patches on that image
        new_shuffled_image = torch.clone(image_tensor)
        for idx, box in zip(permutable_boxes, shuffled_boxes):
            idx, box = idx.item(), box.item()

            row_target = (idx // self.num_squares[1]) * row_height
            col_target = (idx % self.num_squares[1]) * column_width
            row_image = (box // self.num_squares[1]) * row_height
            col_image = (box % self.num_squares[1]) * column_width

            new_shuffled_image[
                :,
                row_target : row_target + row_height,
                col_target : col_target + column_width,
            ] = image_tensor[
                :,
                row_image : row_image + row_height,
                col_image : col_image + column_width,
            ]

        return new_shuffled_image


if __name__ == "__main__":
    sample_image = Image.open("about.jpg")
    sample_image = ImageOps.exif_transpose(sample_image)
    trans = transforms.Compose(
        [
            transforms.Resize((400, 300)),
            transforms.ToTensor(),
        ]
    )

    inv_trans = transforms.Compose([transforms.ToPILImage()])

    ps = PatchShuffle((10, 25), (400, 300))

    tim = trans(sample_image)
    
    # Takes ~1.4s on my machine
    # res = timeit.timeit(lambda: ps(tim, torch.tensor([[110, 190], [250, 290]])), number=1000)
    # print(res)
    
    res = ps(tim, torch.tensor([[110, 190], [250, 290]]))
    res_final = inv_trans(res)
    res_final.save("res.jpg")