import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load a pre-trained VGG16 network
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        # Extract features from the first few layers (captures texture and edges)
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16]).eval()

        # Freeze VGG weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _normalize_batch(self, images):
        images = torch.clamp(images, 0.0, 1.0)
        return self.normalize(images)

    def forward(self, enhanced_img, ground_truth_img):
        # Calculate loss based on deep feature maps rather than raw pixels
        enhanced_features = self.feature_extractor(self._normalize_batch(enhanced_img))
        gt_features = self.feature_extractor(self._normalize_batch(ground_truth_img))
        return self.criterion(enhanced_features, gt_features)


class FusionLoss(nn.Module):
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1):
        super(FusionLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, enhanced_img, ground_truth_img):
        loss_l1 = self.l1_loss(enhanced_img, ground_truth_img)
        loss_perceptual = self.perceptual_loss(enhanced_img, ground_truth_img)
        total_loss = (self.l1_weight * loss_l1) + (self.perceptual_weight * loss_perceptual)
        return total_loss