from drema.scene.cameras import Camera


class DepthCamera(Camera):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, data_device, depth):

        super().__init__(colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, data_device=data_device)
        self.depth = depth
