from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict, OmegaConf
from typing import Dict, Any
import os
import torch
import logging
from model.probe import ProbedLlamaForCausalLM

# 尝试导入PEFT库，如果安装了的话
try:
    import peft
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Any] = {}


def _register_model(model_class):
    MODEL_REGISTRY[model_class.__name__] = model_class


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_model(model_cfg: DictConfig):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    torch_dtype = get_dtype(model_args)
    model_handler = model_cfg.get("model_handler", "AutoModelForCausalLM")
    model_cls = MODEL_REGISTRY[model_handler]
    with open_dict(model_args):
        model_path = model_args.pop("pretrained_model_name_or_path", None)
    
    # 检查是否使用PEFT（如LoRA）
    use_peft = model_cfg.get("use_peft", False)
    # 获取PEFT配置并转换为Python原生类型
    peft_config = model_cfg.get("peft_config", None)
    if peft_config is not None:
        # 将OmegaConf字典转换为普通Python字典
        peft_config = {k: (list(v) if hasattr(v, "__iter__") and not isinstance(v, (dict, str)) else v) 
                     for k, v in peft_config.items()}
    
    try:
        # 使用半精度加载模型
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,  # 启用低CPU内存使用模式
            **model_args,
            cache_dir=hf_home,
        )
        
        logger.info(f"Model loaded with torch_dtype={torch_dtype} for memory optimization")
        
        # 如果配置了使用PEFT（LoRA等），应用PEFT配置
        if use_peft and peft_config:
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                
                logger.info(f"Applying PEFT configuration: {peft_config}")
                peft_type = peft_config.get("peft_type", "LORA").upper()
                
                if peft_type == "LORA":
                    # 从OmegaConf获取target_modules并转换为Python原生列表
                    target_modules = peft_config.get("target_modules", None)
                    if target_modules is not None:
                        if hasattr(target_modules, "_is_missing") and target_modules._is_missing():
                            target_modules = None
                        else:
                            target_modules = list(target_modules)
                    
                    # 创建LoRA配置
                    lora_config = LoraConfig(
                        r=int(peft_config.get("r", 8)),
                        lora_alpha=int(peft_config.get("lora_alpha", 16)),
                        lora_dropout=float(peft_config.get("lora_dropout", 0.05)),
                        bias=peft_config.get("bias", "none"),
                        task_type=getattr(TaskType, peft_config.get("task_type", "CAUSAL_LM")),
                        target_modules=target_modules,
                    )
                    
                    # 应用LoRA配置到模型
                    model = get_peft_model(model, lora_config)
                    model.print_trainable_parameters()  # 打印可训练参数信息
                    
                    # 确保LoRA参数需要梯度
                    for name, param in model.named_parameters():
                        if "lora_" in name.lower():
                            param.requires_grad = True
                            logger.info(f"Set {name} requires_grad=True")
                    
                    # 验证有可训练参数
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    if trainable_params == 0:
                        logger.error("No trainable parameters found! LoRA may not be properly configured.")
                        raise ValueError("No trainable parameters in the model")
                    else:
                        logger.info(f"Total trainable parameters: {trainable_params:,}")
                    
                    # 验证模型精度
                    sample_param = next(model.parameters())
                    logger.info(f"Model precision after LoRA: {sample_param.dtype}")
                    logger.info(f"Model device: {sample_param.device}")
                    
                    # 确保LoRA适配器也是正确的精度
                    for name, param in model.named_parameters():
                        if "lora_" in name.lower() and param.requires_grad:
                            if param.dtype != torch_dtype:
                                logger.warning(f"LoRA parameter {name} has dtype {param.dtype}, converting to {torch_dtype}")
                                param.data = param.data.to(dtype=torch_dtype)
                else:
                    logger.warning(f"Unsupported PEFT type: {peft_type}. Currently only LORA is supported.")
            except ImportError:
                logger.error("PEFT library not installed. Please install it with: pip install peft")
                raise
    except Exception as e:
        logger.warning(f"Model {model_path} requested with {model_cfg.model_args}")
        raise ValueError(
            f"Error {e} while fetching model using {model_handler}.from_pretrained()."
        )
    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        # Convert OmegaConf to plain dict for HF compatibility
        tokenizer_kwargs = OmegaConf.to_container(tokenizer_cfg, resolve=True)
        if not isinstance(tokenizer_kwargs, dict):
            tokenizer_kwargs = dict(tokenizer_cfg)
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config (DictConfig): {tokenizer_cfg}\n"
            f"Converted tokenizer kwargs (dict): {locals().get('tokenizer_kwargs', None)}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer


# register models
_register_model(AutoModelForCausalLM)
_register_model(ProbedLlamaForCausalLM)
