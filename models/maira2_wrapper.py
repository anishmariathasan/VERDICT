"""MAIRA-2 model wrapper with feature extraction capabilities for VERDICT project.

This module provides a comprehensive interface for loading and using the
Microsoft MAIRA-2 model for chest X-ray report generation and analysis.

Architecture:
    - Vision Encoder: Rad-DINO (ViT-B/14) - 518x518 input, 37x37 patches
    - Adapter: 4-layer MLP projection
    - Language Model: Vicuna-7B (32 layers)

Key Details:
    - Image Size: 518x518 (NOT 224x224)
    - Vision Tokens: 1370 per layer (1 CLS + 1369 patches from 37×37 grid)
    - Vision Layers: 12 ViT layers in Rad-DINO encoder
    - Hidden Dimension: 768
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image

try:
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except ImportError:
    AutoModel = None
    AutoModelForCausalLM = None
    AutoProcessor = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

logger = logging.getLogger(__name__)


# ============================================================================
# Constants for MAIRA-2 Architecture
# ============================================================================

MAIRA2_IMAGE_SIZE = 518  # CRITICAL: MAIRA-2 uses 518x518, NOT 224x224
MAIRA2_PATCH_SIZE = 14
MAIRA2_NUM_PATCHES = 37  # 518 / 14 = 37 patches per dimension
MAIRA2_NUM_VISION_TOKENS = 1370  # 1 CLS + 37*37 = 1 + 1369 patch tokens
MAIRA2_VISION_HIDDEN_DIM = 768
MAIRA2_NUM_VISION_LAYERS = 12
MAIRA2_LLM_HIDDEN_DIM = 4096  # Vicuna-7B hidden dimension
MAIRA2_NUM_LLM_LAYERS = 32


# ============================================================================
# Output Dataclasses
# ============================================================================

@dataclass
class MAIRA2Output:
    """
    Output from MAIRA-2 model generation.
    
    Attributes:
        generated_text: The generated report text.
        generated_ids: Token IDs of the generated sequence.
        vision_features: Dictionary of vision features by layer index.
            Shape per layer: [batch_size, num_tokens, hidden_dim]
            For MAIRA-2: [B, 1370, 768]
        vision_tokens: Final vision token embeddings after projection.
        attention_maps: Dictionary of attention maps by layer/type.
        cross_attention_weights: Cross-attention from language to vision tokens.
    """
    generated_text: str
    generated_ids: torch.Tensor
    vision_features: Optional[Dict[int, torch.Tensor]] = None
    vision_tokens: Optional[torch.Tensor] = None
    attention_maps: Optional[Dict[str, torch.Tensor]] = None
    cross_attention_weights: Optional[torch.Tensor] = None


@dataclass
class MAIRA2Config:
    """
    Configuration for MAIRA-2 model loading and inference.
    
    Attributes:
        checkpoint: HuggingFace model checkpoint path.
        device: Device to load model on ('cuda', 'cpu', or 'auto').
        load_in_8bit: Whether to use 8-bit quantisation for memory efficiency.
        load_in_4bit: Whether to use 4-bit quantisation.
        use_flash_attention: Whether to use Flash Attention 2.
        trust_remote_code: Whether to trust remote code from HuggingFace.
        max_new_tokens: Maximum tokens to generate.
        num_beams: Number of beams for beam search.
        temperature: Sampling temperature (0 for greedy).
        top_p: Nucleus sampling probability.
        do_sample: Whether to use sampling (False for greedy decoding).
    """
    checkpoint: str = "microsoft/maira-2"
    device: str = "cuda"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = True
    trust_remote_code: bool = True
    max_new_tokens: int = 512
    num_beams: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    gradient_checkpointing: bool = False
    torch_dtype: str = "float16"
    
    # Prompt templates for different tasks
    prompts: Dict[str, str] = field(default_factory=lambda: {
        "findings": "Describe the findings in this chest X-ray:",
        "impression": "Provide a clinical impression for this chest X-ray:",
        "full_report": "Generate a complete radiology report for this chest X-ray:",
        "vqa": "Question: {question}\nAnswer:",
    })


# ============================================================================
# Main Model Wrapper
# ============================================================================

class MAIRA2Model(nn.Module):
    """
    Wrapper for Microsoft MAIRA-2 vision-language model.
    
    Provides a unified interface for loading, inference, and feature extraction
    from MAIRA-2 for radiology report generation and attribution analysis.
    
    Key Features:
        - Vision token extraction from all 12 ViT layers
        - Attention map extraction for interpretability
        - Batch processing support
        - Memory-efficient inference with quantisation
        - Feature hooks for attribution methods (CoIBA, etc.)
    
    Architecture Details:
        - Input: 518x518 images (NOT 224x224!)
        - Vision tokens: 1370 per layer (1 CLS + 1369 patch tokens from 37×37 grid)
        - Preprocessing: B-spline interpolation, [0, 255] normalisation
    
    Attributes:
        config: MAIRA2Config instance.
        model: The underlying HuggingFace model.
        processor: The model processor for image/text preprocessing.
        tokenizer: The tokenizer for text processing.
    
    Example:
        >>> model = MAIRA2Model.from_pretrained("microsoft/maira-2")
        >>> output = model.generate_report(image, extract_features=True)
        >>> print(output.generated_text)
        >>> print(f"Vision features shape: {output.vision_features[0].shape}")
    """
    
    def __init__(self, config: Optional[MAIRA2Config] = None) -> None:
        """
        Initialise MAIRA-2 model wrapper.
        
        Args:
            config: Model configuration. If None, uses defaults.
        """
        super().__init__()
        self.config = config or MAIRA2Config()
        
        # Model components (initialised in load_model)
        self.model: Optional[nn.Module] = None
        self.processor = None
        self.tokenizer = None
        
        # Component references (set after loading)
        self._vision_encoder: Optional[nn.Module] = None
        self._language_model: Optional[nn.Module] = None
        
        # Feature extraction hooks
        self._vision_features: Dict[int, torch.Tensor] = {}
        self._attention_maps: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # State tracking
        self._is_loaded = False
    
    # ========================================================================
    # Class Methods for Loading
    # ========================================================================
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "microsoft/maira-2",
        device: str = "cuda",
        load_in_8bit: bool = False,
        use_flash_attention: bool = True,
        **kwargs,
    ) -> "MAIRA2Model":
        """
        Load MAIRA-2 model from HuggingFace checkpoint.
        
        Args:
            checkpoint: HuggingFace model checkpoint.
            device: Device to load model on.
            load_in_8bit: Whether to use 8-bit quantisation.
            use_flash_attention: Whether to use Flash Attention 2.
            **kwargs: Additional configuration options.
        
        Returns:
            Loaded MAIRA2Model instance.
        
        Example:
            >>> model = MAIRA2Model.from_pretrained(
            ...     "microsoft/maira-2",
            ...     device="cuda",
            ...     load_in_8bit=True
            ... )
        """
        config = MAIRA2Config(
            checkpoint=checkpoint,
            device=device,
            load_in_8bit=load_in_8bit,
            use_flash_attention=use_flash_attention,
            **kwargs
        )
        instance = cls(config)
        instance.load_model()
        return instance
    
    # ========================================================================
    # Model Loading
    # ========================================================================
    
    def load_model(self) -> None:
        """
        Load the model and processor from checkpoint.
        
        Raises:
            ImportError: If transformers is not installed.
            RuntimeError: If model loading fails.
        """
        if AutoModelForCausalLM is None:
            raise ImportError(
                "transformers is required for MAIRA-2. "
                "Install with: pip install transformers>=4.35.0"
            )
        
        logger.info(f"Loading MAIRA-2 from {self.config.checkpoint}")
        
        # Configure quantisation
        quantization_config = None
        if self.config.load_in_8bit or self.config.load_in_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "bitsandbytes is required for quantisation. "
                    "Install with: pip install bitsandbytes"
                )
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.config.load_in_8bit,
                load_in_4bit=self.config.load_in_4bit,
            )
        
        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)
        
        # Build model loading kwargs
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch_dtype,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = self.config.device
        
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.checkpoint,
                **model_kwargs,
            )
            
            # Enable gradient checkpointing if requested
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
        
        # Load processor and tokenizer
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.checkpoint,
                trust_remote_code=self.config.trust_remote_code,
            )
            self.tokenizer = self.processor.tokenizer
        except Exception as e:
            logger.warning(f"Processor not found, trying tokenizer only: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.checkpoint,
                trust_remote_code=self.config.trust_remote_code,
            )
        
        # Identify model components
        self._vision_encoder = self._get_vision_encoder()
        self._language_model = self._get_language_model()
        
        self._is_loaded = True
        self.model.eval()
        
        logger.info("MAIRA-2 loaded successfully")
        logger.info(f"  - Vision encoder: {type(self._vision_encoder).__name__}")
        logger.info(f"  - Language model: {type(self._language_model).__name__}")
    
    def _get_vision_encoder(self) -> nn.Module:
        """
        Get vision encoder module (Rad-DINO ViT).
        
        Returns:
            The vision encoder module.
        
        Raises:
            AttributeError: If vision encoder cannot be found.
        """
        # MAIRA-2 architecture variants
        if hasattr(self.model, 'vision_tower'):
            return self.model.vision_tower
        elif hasattr(self.model, 'vision_model'):
            return self.model.vision_model
        elif hasattr(self.model, 'vision_encoder'):
            return self.model.vision_encoder
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_tower'):
            return self.model.model.vision_tower
        else:
            raise AttributeError(
                "Cannot find vision encoder in MAIRA-2 model. "
                "Expected attributes: vision_tower, vision_model, or vision_encoder"
            )
    
    def _get_language_model(self) -> nn.Module:
        """
        Get language model module (Vicuna).
        
        Returns:
            The language model module.
        
        Raises:
            AttributeError: If language model cannot be found.
        """
        if hasattr(self.model, 'language_model'):
            return self.model.language_model
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model
        elif hasattr(self.model, 'lm_head'):
            return self.model
        else:
            raise AttributeError(
                "Cannot find language model in MAIRA-2 model. "
                "Expected attributes: language_model or model"
            )
    
    # ========================================================================
    # Image Preprocessing
    # ========================================================================
    
    def preprocess_image(
        self,
        image: Union[Image.Image, torch.Tensor, List[Image.Image]],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess image(s) for MAIRA-2.
        
        CRITICAL: MAIRA-2 uses 518x518 resolution, NOT 224x224!
        Preprocessing includes B-spline interpolation and [0, 255] normalisation.
        
        Args:
            image: Input image (PIL Image, tensor, or list of images).
            return_tensors: Format to return ("pt" for PyTorch).
        
        Returns:
            Preprocessed inputs dictionary with pixel_values.
            Shape: [batch_size, 3, 518, 518]
        """
        if self.processor is not None:
            inputs = self.processor(
                images=image,
                return_tensors=return_tensors,
            )
        else:
            # Manual preprocessing fallback
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.Resize((MAIRA2_IMAGE_SIZE, MAIRA2_IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            
            if isinstance(image, list):
                pixel_values = torch.stack([transform(img) for img in image])
            else:
                pixel_values = transform(image).unsqueeze(0)
            
            inputs = {"pixel_values": pixel_values}
        
        # Move to device
        device = self.config.device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        
        return inputs
    
    # ========================================================================
    # Feature Extraction Hooks
    # ========================================================================
    
    def register_feature_hooks(self) -> None:
        """
        Register forward hooks to extract vision features and attention maps.
        
        Extracts:
            - Vision tokens from all 12 ViT layers
              Shape: [batch_size, 1370, 768] (1 CLS + 1369 patches)
            - Attention maps from language model cross-attention
        """
        self._remove_hooks()
        
        # Find vision encoder layers
        vision_layers = self._find_vision_layers()
        
        for layer_idx, layer in enumerate(vision_layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    # Extract hidden states from layer output
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    
                    # Store detached features
                    self._vision_features[idx] = hidden_states.detach().cpu()
                
                return hook_fn
            
            handle = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)
        
        logger.debug(f"Registered {len(self._hooks)} feature extraction hooks")
    
    def _find_vision_layers(self) -> List[nn.Module]:
        """
        Find vision encoder transformer layers.
        
        Returns:
            List of vision transformer layer modules.
        """
        vision_encoder = self._vision_encoder
        
        # Common attribute names for ViT layers
        layer_attrs = [
            'encoder.layer',      # HuggingFace ViT
            'blocks',             # timm ViT
            'layers',             # Generic
            'encoder.layers',
        ]
        
        for attr_path in layer_attrs:
            try:
                obj = vision_encoder
                for attr in attr_path.split('.'):
                    obj = getattr(obj, attr)
                return list(obj)
            except AttributeError:
                continue
        
        # Fallback: search for Sequential or ModuleList
        for name, module in vision_encoder.named_modules():
            if isinstance(module, (nn.Sequential, nn.ModuleList)) and len(module) > 1:
                return list(module)
        
        logger.warning("Could not find vision layers, returning empty list")
        return []
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._vision_features = {}
        self._attention_maps = {}
    
    # ========================================================================
    # Report Generation
    # ========================================================================
    
    @torch.no_grad()
    def generate_report(
        self,
        image: Union[Image.Image, torch.Tensor],
        prompt: Optional[str] = None,
        prompt_type: str = "findings",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extract_features: bool = False,
    ) -> MAIRA2Output:
        """
        Generate radiology report for chest X-ray image.
        
        Args:
            image: Input chest X-ray image (PIL Image or tensor).
            prompt: Custom text prompt. If None, uses prompt_type.
            prompt_type: Type of default prompt ('findings', 'impression', etc.).
            max_new_tokens: Maximum tokens to generate (default from config).
            temperature: Sampling temperature (default from config).
            top_p: Nucleus sampling parameter (default from config).
            extract_features: Whether to extract vision features via hooks.
        
        Returns:
            MAIRA2Output with generated text and optional features.
        
        Example:
            >>> output = model.generate_report(
            ...     image,
            ...     prompt="Describe any abnormalities:",
            ...     extract_features=True
            ... )
            >>> print(output.generated_text)
            >>> for layer, feat in output.vision_features.items():
            ...     print(f"Layer {layer}: {feat.shape}")
        """
        self._check_loaded()
        
        # Register hooks if feature extraction requested
        if extract_features:
            self.register_feature_hooks()
        
        # Get prompt
        if prompt is None:
            prompt = self.config.prompts.get(prompt_type, self.config.prompts["findings"])
        
        # Preprocess image
        image_inputs = self.preprocess_image(image)
        
        # Tokenize prompt
        if self.processor is not None:
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        else:
            text_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.config.device)
            inputs = {**image_inputs, **text_inputs}
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": top_p or self.config.top_p,
            "do_sample": self.config.do_sample and (temperature or self.config.temperature) > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Generate
        outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode generated text
        if self.processor is not None:
            generated_text = self.processor.decode(
                outputs[0],
                skip_special_tokens=True,
            )
        else:
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )
        
        # Remove prompt from output if present
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()
        
        # Prepare output
        result = MAIRA2Output(
            generated_text=generated_text,
            generated_ids=outputs,
        )
        
        # Add extracted features if requested
        if extract_features:
            result.vision_features = self._vision_features.copy()
            result.attention_maps = self._attention_maps.copy()
            self._remove_hooks()
        
        return result
    
    @torch.no_grad()
    def get_vision_features(
        self,
        image: Union[Image.Image, torch.Tensor],
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract vision features from specific layers.
        
        Args:
            image: Input image.
            layers: Layer indices to extract (None = all 12 layers).
        
        Returns:
            Dictionary mapping layer index to features.
            Features shape: [batch_size, 1370, 768]
            (1 CLS token + 1369 patch tokens from 37×37 grid)
        
        Example:
            >>> features = model.get_vision_features(image, layers=[0, 6, 11])
            >>> for layer, feat in features.items():
            ...     print(f"Layer {layer}: {feat.shape}")
            # Layer 0: torch.Size([1, 1370, 768])
            # Layer 6: torch.Size([1, 1370, 768])
            # Layer 11: torch.Size([1, 1370, 768])
        """
        self._check_loaded()
        self.register_feature_hooks()
        
        # Forward pass through vision encoder only
        inputs = self.preprocess_image(image)
        
        # Run vision encoder forward
        with torch.no_grad():
            if hasattr(self._vision_encoder, 'forward'):
                _ = self._vision_encoder(inputs.get('pixel_values'))
            else:
                # Try full model forward if vision encoder doesn't work standalone
                _ = self.model(**inputs, output_hidden_states=True)
        
        # Filter requested layers
        if layers is not None:
            features = {
                k: v for k, v in self._vision_features.items() 
                if k in layers
            }
        else:
            features = self._vision_features.copy()
        
        self._remove_hooks()
        
        return features
    
    @torch.no_grad()
    def batch_generate_reports(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        prompts: Optional[List[str]] = None,
        batch_size: int = 4,
        **generation_kwargs,
    ) -> List[MAIRA2Output]:
        """
        Generate reports for a batch of images.
        
        Args:
            images: List of input images.
            prompts: Optional list of prompts (one per image).
            batch_size: Batch size for processing.
            **generation_kwargs: Additional generation arguments.
        
        Returns:
            List of MAIRA2Output objects.
        """
        if prompts is None:
            prompts = [None] * len(images)
        
        assert len(images) == len(prompts), \
            f"Number of images ({len(images)}) must match prompts ({len(prompts)})"
        
        outputs = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]
            
            for img, prompt in zip(batch_images, batch_prompts):
                output = self.generate_report(img, prompt=prompt, **generation_kwargs)
                outputs.append(output)
        
        return outputs
    
    # ========================================================================
    # Model Forward Pass
    # ========================================================================
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            images: Image tensor of shape (B, C, H, W). H, W should be 518.
            input_ids: Token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            labels: Label token IDs for training.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states.
            return_dict: Whether to return a dictionary.
        
        Returns:
            Model outputs including logits, loss, attentions, etc.
        """
        self._check_loaded()
        
        outputs = self.model(
            pixel_values=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        return outputs
    
    # ========================================================================
    # Component Accessors
    # ========================================================================
    
    def get_vision_encoder(self) -> nn.Module:
        """
        Get the vision encoder component (Rad-DINO ViT).
        
        Returns:
            Vision encoder module.
        """
        self._check_loaded()
        return self._vision_encoder
    
    def get_language_model(self) -> nn.Module:
        """
        Get the language model component (Vicuna-7B).
        
        Returns:
            Language model module.
        """
        self._check_loaded()
        return self._language_model
    
    def get_vision_projection(self) -> Optional[nn.Module]:
        """
        Get the vision-to-language projection layer (4-layer MLP adapter).
        
        Returns:
            Projection module if found, None otherwise.
        """
        self._check_loaded()
        
        # Common attribute names for projection layer
        projection_attrs = [
            'multi_modal_projector',
            'mm_projector',
            'vision_projection',
            'adapter',
        ]
        
        for attr in projection_attrs:
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
        
        return None
    
    # ========================================================================
    # Model State Management
    # ========================================================================
    
    def freeze_vision_encoder(self) -> None:
        """Freeze vision encoder parameters for fine-tuning."""
        self._check_loaded()
        
        for param in self._vision_encoder.parameters():
            param.requires_grad = False
        
        logger.info("Vision encoder frozen")
    
    def freeze_language_model(self) -> None:
        """Freeze language model parameters for fine-tuning."""
        self._check_loaded()
        
        for param in self._language_model.parameters():
            param.requires_grad = False
        
        logger.info("Language model frozen")
    
    def unfreeze_all(self) -> None:
        """Unfreeze all model parameters."""
        self._check_loaded()
        
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info("All parameters unfrozen")
    
    def _check_loaded(self) -> None:
        """Check that model is loaded."""
        if not self._is_loaded:
            raise RuntimeError(
                "Model not loaded. Call load_model() first or use from_pretrained()."
            )
    
    def to(self, device: Union[str, torch.device]) -> "MAIRA2Model":
        """Move model to device."""
        if self.model is not None:
            self.model = self.model.to(device)
        self.config.device = str(device)
        return self
    
    def eval(self) -> "MAIRA2Model":
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
        return self
    
    def train(self, mode: bool = True) -> "MAIRA2Model":
        """Set model training mode."""
        if self.model is not None:
            self.model.train(mode)
        return self
    
    @property
    def device(self) -> str:
        """Get the device the model is on."""
        return self.config.device
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"MAIRA2Model(\n"
            f"  checkpoint={self.config.checkpoint},\n"
            f"  device={self.config.device},\n"
            f"  loaded={self._is_loaded},\n"
            f"  image_size={MAIRA2_IMAGE_SIZE}x{MAIRA2_IMAGE_SIZE},\n"
            f"  vision_tokens={MAIRA2_NUM_VISION_TOKENS},\n"
            f"  vision_layers={MAIRA2_NUM_VISION_LAYERS}\n"
            f")"
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Initialise model
    print("Loading MAIRA-2...")
    model = MAIRA2Model.from_pretrained(
        "microsoft/maira-2",
        device="cuda",
        load_in_8bit=True,  # Use quantisation for memory efficiency
    )
    
    print(model)
    
    # Load test image
    image = Image.new('RGB', (512, 512), color='gray')  # Dummy image for testing
    
    # Generate report with feature extraction
    print("\nGenerating report...")
    output = model.generate_report(
        image,
        prompt="Describe the findings in this chest X-ray:",
        extract_features=True
    )
    
    print("\nGenerated Report:")
    print(output.generated_text)
    
    if output.vision_features:
        print("\nExtracted Vision Features:")
        for layer_idx, features in sorted(output.vision_features.items()):
            print(f"  Layer {layer_idx}: {features.shape}")
            # Expected: torch.Size([1, 1370, 768])
    
    # Extract features from specific layers
    print("\nExtracting features from layers [0, 6, 11]...")
    features = model.get_vision_features(image, layers=[0, 6, 11])
    for layer_idx, feat in features.items():
        print(f"  Layer {layer_idx}: {feat.shape}")
