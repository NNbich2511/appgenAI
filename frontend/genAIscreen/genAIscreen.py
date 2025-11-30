from kivy.uix.screenmanager import Screen
from transformers import CLIPTokenizer
import torch
import threading
from kivy.clock import mainthread
from genAI import pipeline, genAI


class genAIscreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Thiết lập device
        self.DEVICE = "cpu"
        ALLOW_CUDA = True
        ALLOW_MPS = False

        if torch.cuda.is_available() and ALLOW_CUDA:
            self.DEVICE = "cuda"
        elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
            self.DEVICE = "mps"
        print(f"Using device: {self.DEVICE}")

        # Tokenizer
        self.tokenizer = CLIPTokenizer(
            "D://openaiclip-vit-base-patch16//vocab.json",
            merges_file="D://openaiclip-vit-base-patch16//merges.txt"
        )

        # Load model một lần khi init
        model_file = "D://v1-5-pruned-emaonly.ckpt"
        self.models = genAI.preload_models_from_standard_weights(model_file, self.DEVICE)

    def create(self):
        """Khởi tạo thread để tạo ảnh"""
        prompt = self.ids.promt.text
        if not prompt:
            self._set_status("Prompt trống!")
            return

        self._set_status("Đang tạo ảnh...")
        threading.Thread(target=self._generate_image, args=(prompt,), daemon=True).start()

    def _generate_image(self, prompt):
        """Hàm chạy ngoài thread"""
        output_image = pipeline.generate(
            prompt=prompt,
            uncond_prompt="",
            input_image=None,
            strength=0.9,
            do_cfg=True,
            cfg_scale=8,
            sampler_name="ddpm",
            n_inference_steps=50,
            seed=42,
            models=self.models,
            device=self.DEVICE,
            idle_device="cpu",
            tokenizer=self.tokenizer,
        )

        # Cập nhật GUI trên main thread
        self._update_image(output_image)

    @mainthread
    def _update_image(self, output_image):
        output_image.save("output.png")
        self.ids.output_img.source = "output.png"
        self.ids.output_img.reload()
        self._set_status("Hoàn tất!")

    @mainthread
    def _set_status(self, text):
        self.ids.status_label.text = text

    def logscreen(self):
        """Xử lý khi người dùng nhấn nút đăng xuất"""
        app = self.manager.app  # Lấy thể hiện của App
        self.manager.current = "home"  # Quay về màn hình login
        login_screen =  self.manager.get_screen("home")