import numpy as np
import torch
from PIL import Image

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper


NUM_CAM_FRAMES = 4
NUM_CAM_SLOTS  = 4
IMG_H, IMG_W   = 1080, 1920


class AlpamayoInference:
    def __init__(self, device="cuda"):
        self.device = device

        print(f"[Alpamayo] loading model on {self.device}...")

        self.model = AlpamayoR1.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            dtype=torch.bfloat16
        ).to(self.device)

        self.model.eval()
        self.processor = helper.get_processor(self.model.tokenizer)

        print("[Alpamayo] ready")

    def _prepare_image(self, image_path):
        # image_frames shape expected by create_message: [NUM_CAM_FRAMES * NUM_CAM_SLOTS, 3, H, W]
        image_frames = torch.zeros(
            NUM_CAM_FRAMES, NUM_CAM_SLOTS, 3, IMG_H, IMG_W,
            dtype=torch.uint8
        )

        img = Image.open(image_path).convert("RGB").resize((IMG_W, IMG_H))
        img_t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)
        image_frames[NUM_CAM_FRAMES - 1, 0] = img_t

        return image_frames.flatten(0, 1)  # [NUM_CAM_FRAMES * NUM_CAM_SLOTS, 3, H, W]

    def predict(self, image_path, ego_xyz, ego_rot,
                num_traj=6, seed=42):
        """
        Returns:
            pred_xyz: (K, T, 3)  — matches test_inference output sliced to [0, 0]
        """
        image_frames = self._prepare_image(image_path)

        messages = helper.create_message(image_frames)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": torch.from_numpy(ego_xyz).float().unsqueeze(0).unsqueeze(0),
            "ego_history_rot": torch.from_numpy(ego_rot).float().unsqueeze(0).unsqueeze(0),
        }

        model_inputs = helper.to_device(model_inputs, self.device)

        torch.cuda.manual_seed_all(seed)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=num_traj,
                max_generation_length=256,
                return_extra=True,
            )

        # test_inference: pred_xyz is [batch, num_traj_sets, K, T, 3]
        # slice [0, 0] → (K, T, 3)
        return pred_xyz.cpu().numpy()[0, 0]